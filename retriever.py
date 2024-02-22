import faiss, logging, time, tqdm, os, pathlib
from sentence_transformers import SentenceTransformer
import argparse, pickle
import pandas as pd
import numpy as np

    
def get_logger():
    from telegram_logging import TelegramHandler, TelegramFormatter
    logger = logging.getLogger(__name__)
    logging.basicConfig(
            format='[%(levelname)s] (%(asctime)s): %(message)s',
            level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    logger.info('Started')
    return logger

logger = get_logger()

def read_indexes(args, lookup_file):
    indexes = {}
    shards_paths = lookup_file['shards']
    resources = [faiss.StandardGpuResources() for i in range(1)]
    for segment, path in tqdm.tqdm(shards_paths.items(), desc='Reading shards'):
        index = faiss.read_index(path)
        index = faiss.index_cpu_to_gpu_multiple_py(resources=resources, index=index)
        indexes[segment] = index
            
    return indexes


def read_model(args):

    logger.info(f'Reading the model {args.model}')
    model_name = args.model
    model = SentenceTransformer(model_name)

    return model


def get_doc_ids(neighbours, mapping):
    result = []
    for n in neighbours:
        result.append(mapping[n])
    return result


def save_run_trec_file(chunks_result, args, chunk_it):
    output_path = args.output + f'run_{chunk_it}.trec'
    logger.info(f'Saving {output_path}')

    faiss_run = pd.DataFrame(chunks_result, columns=['qid', 'd1', 'docno', 'rank', 'score', 'd2'])
    faiss_run = faiss_run.sort_values(by=['qid', 'score'], ascending=[True, False])
    faiss_run = faiss_run.groupby('qid')
    faiss_run = faiss_run.head(args.hits).reset_index()
    # Rank all retrieved documents
    faiss_run['rank'] = faiss_run.groupby('qid').cumcount() + 1
    
    faiss_run = faiss_run[['qid', 'd1', 'docno', 'rank', 'score', 'd2']]
    faiss_run.to_csv(output_path, index=False, header=False, sep='\t')
    logger.info(f'Saved {output_path}')


def retrieval(args, indexes, lookup_file):
    chunk_size = args.chunks_size
    data = pd.read_csv(args.topics, names=['qid', 'q'], sep='\t')
    number_of_chunks = len(data) // chunk_size
    number_of_chunks = number_of_chunks if number_of_chunks else 1
    chuncked_data = np.array_split(data, number_of_chunks)
    logger.info(f'Total number of shards: {len(chuncked_data)}')
    model = read_model(args)
    
    for chunk_it, chunk in enumerate(chuncked_data):
        chunks_result = []
        encoded_passages = model.encode(chunk['q'].values, 
                                convert_to_numpy=True, 
                                show_progress_bar=True, 
                                batch_size=128,
                                device=args.device)
        qids = chunk['qid'].values
        for segment, index in indexes.items():
            scores, neighbours = index.search(encoded_passages, args.hits)
            
            # Creating run file for this chunk. However, it must filter again for shards
            for i in tqdm.tqdm(range(len(qids)), desc=f'Chunk [{chunk_it}], Seg {segment}'):
                actual_ids = get_doc_ids(neighbours=neighbours[i], 
                            mapping=lookup_file['ids'][segment])
                for j in range(args.hits):
                    chunks_result.append(
                        [qids[i], 'Q0', actual_ids[j], j, scores[i][j], 1]
                    )
        save_run_trec_file(chunks_result, args, chunk_it)
        # Reduce Ram usage
        del scores
        del neighbours
        del qids
        del chunks_result
                                
    return True     
                   

def main():

    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, 
                        default="sentence-transformers/all-MiniLM-L6-v2", 
                        help='Language model name to encode queries with')
    parser.add_argument('--lookup-file', type=str, 
                        default='content/lookup_file.pk',
                        help='The look up file from index step (full path)')
    parser.add_argument('--topics', type=str, 
                        help='Path of topics to be retrieved (TREC format)')
    parser.add_argument('--chunks-size', type=int, default=500_000,
                        help='Size of each chunk. (More size uses more RAM)') 
    parser.add_argument('--device', type=str, 
                        default='cuda:0',
                        help='GPU device (can be set by pytorch too. default cuda:0)') 
    parser.add_argument('--hits', type=int,
                        help='Number of hits for each query')
    parser.add_argument('--output', type=str,
                        help='Output path')
    args = parser.parse_args()

    with open(args.lookup_file, 'rb') as f:
        lookup_file = pickle.load(f)
        
    indexes = read_indexes(args, lookup_file)
    if not args.output:
        raise Exception('Please enter the output file')
    if args.output[-1] != '/':
         args.output = args.output + '/'
    
    pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)

    retrieval(args, indexes, lookup_file)

    logger.info('Finishing the retrieving')
    
    
    logger.info('Finished {} min'.format( (time.time() - start) / 60))
    
        

if __name__ == "__main__":
    main()
