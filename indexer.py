from sentence_transformers import SentenceTransformer
import argparse
import faiss
import logging, argparse, time, tqdm, pickle, pathlib
import pandas as pd
import numpy as np

def get_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(levelname)s] (%(asctime)s): %(message)s',
            level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    logger.info('Started')
    return logger

logger = get_logger()


def read_model(args):

    logger.info(f'Reading the model {args.model}')
    model_name = args.model
    model = SentenceTransformer(model_name)

    return model


def index_sbert(args):
    chunk_size = args.chunks_size
    model = read_model(args)
    logger.info('Reading the input file')
    data = pd.read_csv(args.documents, names=['docno', 'doc'], sep='\t')
    number_of_chunks = len(data) // chunk_size
    number_of_chunks = number_of_chunks if number_of_chunks else 1
    chuncked_data = np.array_split(data, number_of_chunks)
    shards_ids_mapping = {'ids': {}, 'shards': {}}
    logger.info(f'Total number of shards: {len(chuncked_data)}')
    for segment, chunk in tqdm.tqdm(enumerate(chuncked_data), desc='Processing chunks'):
        # print(chunk)
        encoded_passages = model.encode(chunk['doc'].values, 
                                convert_to_numpy=True, 
                                show_progress_bar=True, 
                                batch_size=256,
                                device=args.device)
        dimension = model.get_sentence_embedding_dimension()
        cpu_index = faiss.IndexFlatIP(dimension)
        cpu_index.add(encoded_passages)
        index_path = args.output + f'shard_{segment}.faiss'
        faiss.write_index(cpu_index, index_path)
        
        local_ids = [i for i in range(len(chunk['doc'].values))]
        chunk_ids = chunk['docno'].values
        shards_ids_mapping['ids'][segment] = dict(zip(local_ids, chunk_ids))
        shards_ids_mapping['shards'][segment] = index_path
        
    logger.info('Indexing has been finished.')
    with open(args.lookup_file, 'wb') as f:
        pickle.dump(shards_ids_mapping, f)
    
    logger.info('All files has been saved.')
        

# Run the program as main file
def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help='Language model name to encode queries with')
    parser.add_argument('--documents', type=str,
                        help='Full path of documents (TREC type)') #path to queries we want to index (TSV format)
    parser.add_argument('--output', type=str, 
                        default='content/',
                        help='Path to the saved indexes and shards.') 
    parser.add_argument('--chunks-size', type=int, default=500_000,
                        help='Size of each chunk. (More size uses more RAM)') 
    parser.add_argument('--device', type=str, 
                        default='cuda:0',
                        help='GPU device (can be set by pytorch too. default cuda:0)')
    args = parser.parse_args()
    
    if not args.output[0] != '/':
        args.output = args.output + '/'
    
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
    
    args.lookup_file = args.output + 'lookup_file.pk'

    index_sbert(args)

    logger.info('Finished.\nElasped Time: {}'.format(time.time()-start))


if __name__ == "__main__":
    main()