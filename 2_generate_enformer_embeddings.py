from pyfaidx import Fasta
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from enformer_pytorch import Enformer
from tqdm import tqdm
import pickle, math
import h5py, os, glob
from collections import defaultdict
from enformer_pytorch import from_pretrained

data_dir = 'data/'
device = 'cuda'
dna_lkp = {'A':0,'C':1,'G':2,'T':3,'N':4}
enformer_window  = 128
enformer_embedding_shape = 3072

def seq_encoder(dna_str):
    encoded_seq = []
    for s in dna_str:
        encoded_seq.append(dna_lkp[s.upper()])
    return torch.tensor(encoded_seq)

# pytorch dataset - to get the genome data
class Gendataset(Dataset):
    def __init__(self,chr_tiles,genome_fasta):
        
        self.enformer_exclude_ends = 40960
        self.enformer_input_length = 196608
        self.enformer_embed_length = 114688
        self.enformer_one_side = self.enformer_exclude_ends + self.enformer_embed_length
        
        # load the genome data
        self.genome_data = Fasta(genome_fasta,as_raw=True)
        # load the bed file data
        self.regions = chr_tiles

    def __len__(self):
        return len(self.regions)

    def __getitem__(self,idx):
        chr_name   = self.regions[idx][0].strip()
        chr_start  = int(self.regions[idx][1])
        chr_end    = int(self.regions[idx][2])
        total_seq_length = chr_end - chr_start

        # need to handle few extra things for processing thru enformer
        if total_seq_length < self.enformer_input_length:
            if chr_start == 0: # start padding
                padded_seq = ('N' * self.enformer_exclude_ends) + self.genome_data[chr_name][chr_start:chr_end]
                
                if len(padded_seq) != self.enformer_input_length: # can happen to extremely small chr
                    leftover_gap = (self.enformer_input_length - len(padded_seq))
                    padded_seq =  padded_seq + 'N' * leftover_gap
                    chr_embed_start = chr_start + 1
                    chr_embed_end   = chr_end
                else:
                    chr_embed_start = chr_start + 1
                    chr_embed_end   = (chr_embed_start + self.enformer_embed_length) - 1
            else: # tail ends
                add_Ns = self.enformer_input_length - total_seq_length ##### CHANGE 1
                padded_seq = self.genome_data[chr_name][chr_start:chr_end] + ('N' * add_Ns) ##### CHANGE 1
                #padded_seq = self.genome_data[chr_name][chr_start:chr_end] + ('N' * self.enformer_exclude_ends)
                chr_embed_start = chr_start + 1 + self.enformer_exclude_ends 
                chr_embed_end   = chr_end
        else:
            padded_seq = self.genome_data[chr_name][chr_start:chr_end] # pydiax library needs 1 offset
            chr_embed_start = chr_start + 1 + self.enformer_exclude_ends 
            chr_embed_end   = (chr_embed_start + self.enformer_embed_length) - 1
        
        # encode sequence
        assert len(padded_seq) == self.enformer_input_length, "genmore sequence length is not 196608"
        genome_seq = seq_encoder(padded_seq)
            
        return {'chr_name': chr_name,'chr_start':chr_start,'chr_end':chr_end,
                'chr_embed_start':chr_embed_start,'chr_embed_end':chr_embed_end,
                'genome_seq':genome_seq}
    
# enformer model
# model = Enformer.from_hparams(
#     dim = 1536,
#     depth = 11,
#     heads = 8,
#     output_heads = dict(human = 5313, mouse = 1643),
#     target_length = 896,
# ).to(device)

# load weights
model = from_pretrained('EleutherAI/enformer-official-rough').to(device)

# save edge regions for later processing
def save_edge_regions(chr_info,tile_embeddings,chr_region_edges_idx):

    region_start,region_start_1  = int(chr_info[1]) , int(chr_info[1]) + enformer_window
    rel_pos =  math.floor(((int(chr_info[2]) - int(chr_info[1])) / enformer_window ) - 1) # normal case 874
    region_end_1, region_end = region_start + enformer_window * rel_pos , region_start + enformer_window * (rel_pos + 1)

    print(f"Storing edge embeddings for actual position {region_start,region_start_1,region_end_1,region_end}")
    print(f"Storing edge embeddings for relative positions is 0,1,{rel_pos,rel_pos + 1}")

    torch.save(tile_embeddings[0].to('cpu'),f'{data_dir}{str(chr_info[0])}_{str(region_start)}_embeddings.pt')
    chr_region_edges_idx.append(region_start)
    
    torch.save(tile_embeddings[1].to('cpu'),f'{data_dir}{str(chr_info[0])}_{str(region_start_1)}_embeddings.pt')
    chr_region_edges_idx.append(region_start_1)
    
    torch.save(tile_embeddings[rel_pos].to('cpu'),f'{data_dir}{str(chr_info[0])}_{str(region_end_1)}_embeddings.pt')
    chr_region_edges_idx.append(region_end_1)
        
    torch.save(tile_embeddings[rel_pos + 1].to('cpu'),f'{data_dir}{str(chr_info[0])}_{str(region_end)}_embeddings.pt')
    chr_region_edges_idx.append(region_end)

    return chr_region_edges_idx
    
# get enformer embedddings
def get_genome_embeddings(chr_tiles,genome_file,train_region_for_chr,test_region_for_chr,chrs):

    h5_train = h5py.File(f'data/{chrs}_train_enformer.h5', 'a')
    h5_test  = h5py.File(f'data/{chrs}_test_enformer.h5', 'a')
    chr_region_edges_idx = [] # storing edge regions index

    gen_dataloader   = DataLoader(Gendataset(chr_tiles,genome_file), 
                              batch_size = 2,  shuffle = False, pin_memory = False)

    print(f"Starting embeddings processing")

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(gen_dataloader)):

            genome_seq = batch['genome_seq']
            chr_name   = batch['chr_name']
            chr_start  = batch['chr_start']
            chr_end    = batch['chr_end'] 
            chr_embed_start  = batch['chr_embed_start']
            chr_embed_end    = batch['chr_embed_end'] 
            
            # print(f"genome_seq {genome_seq.shape}")
            _, embeddings = model(genome_seq.to(device), return_embeddings = True)
            #print(f"embeddings size {embeddings.shape}")

            for i, chr_info in enumerate(zip(chr_name,chr_embed_start,chr_embed_end,chr_start,chr_end)):
                print(f"chr_name {chr_info[0]},chr_start {chr_info[3]}, chr_end {chr_info[4]},chr_embed_start {chr_info[1] },chr_embed_end {chr_info[2] }")

                # save edges
                chr_region_edges_idx = save_edge_regions(chr_info, embeddings[i,:],chr_region_edges_idx)

                # embeddings for train regions
                print(f"Processing regions from training set")
                if len(train_region_for_chr) != 0:  # only if the region exist in test data
                    train_pos_embedddings_lst,train_region_chr,train_region_pos = \
                        process_region(train_region_for_chr,chr_info,embeddings[i,:])
                
                    if len(train_region_pos) != 0: # only if processed

                        train_region_embedddings_np  = np.array(torch.stack(train_pos_embedddings_lst,axis=0).squeeze(1))
                        train_region_chr_np = np.array(train_region_chr).reshape(-1,1)
                        train_region_pos_np = np.array(train_region_pos).reshape(-1,1)
                        #print(f"region_embedddings_np {region_embedddings_np.shape}")
                        #print(f"chr_np {chr_np.shape}")
                        #print(f"pos_np {pos_np.shape}")

                        # save in the hdf5 file
                        print(f"Saving the data into hdf5 file")
                        save_hdf5(train_region_embedddings_np,train_region_chr_np,train_region_pos_np,h5_train)

                        # remove regions processed
                        print(f"Remove processed regions and update the training set dict")
                        for tp in list(train_region_pos_np[:,0]):
                            #print(f"Removing position {tp}")
                            train_region_for_chr.remove(tp)
                    else:
                        print(f"Regions are all on edge , hence nothing processed")
                else:
                    print(f"Training set does not have this chromosome")

                # embeddings for test regions
                print(f"Processing regions from testing set")
                if len(test_region_for_chr) != 0:  # only if the region exist in test data
                    test_pos_embedddings_lst,test_region_chr,test_region_pos = \
                        process_region(test_region_for_chr,chr_info,embeddings[i,:])
                
                    if len(test_region_pos) != 0: # only if processed

                        test_region_embedddings_np  = np.array(torch.stack(test_pos_embedddings_lst,axis=0).squeeze(1))
                        test_region_chr_np = np.array(test_region_chr).reshape(-1,1)
                        test_region_pos_np = np.array(test_region_pos).reshape(-1,1)
                        
                        # save in the hdf5 file
                        print(f"Saving the data into hdf5 file")
                        save_hdf5(test_region_embedddings_np,test_region_chr_np,test_region_pos_np,h5_test)

                        print(f"Remove processed regions and update the testing set dict")
                        for tp in list(test_region_pos_np[:,0]):
                            #print(f"Removing position {tp}")
                            test_region_for_chr.remove(tp)
                    else:
                        print(f"Regions are all on edge , hence nothing processed")
                else:
                    print(f"Testing set does not have this chromosome")

    h5_train.close()
    h5_test.close()
    return  train_region_for_chr, test_region_for_chr, chr_region_edges_idx

# save the region as hdf5 file
def save_hdf5(region_embedddings_np,region_chr_np,region_pos_np,h5_file):
    if 'embedding' not in h5_file.keys() and 'chr' not in h5_file.keys() and 'pos' not in h5_file.keys():
        print(f"Creating new HDF5 file")
                    
        h5_file.create_dataset(
            'embedding', 
                data=region_embedddings_np, 
                compression="gzip", 
                chunks=True, 
                maxshape=(None,enformer_embedding_shape))
        
        h5_file.create_dataset(
            'chr', 
            data=region_chr_np, 
            compression="gzip", 
            chunks=True, 
            maxshape=(None,1)) 
        
        h5_file.create_dataset(
            'position', 
            data=region_pos_np, 
            compression="gzip", 
            chunks=True, 
            maxshape=(None,1))
    else:
        print(f"Appending to HDF5 file")
    
        h5_file['embedding'].resize((h5_file['embedding'].shape[0] + region_embedddings_np.shape[0]), axis=0)
        h5_file['embedding'][-region_embedddings_np.shape[0]:] = region_embedddings_np

        h5_file['chr'].resize((h5_file['chr'].shape[0] + region_chr_np.shape[0]),axis=0)
        h5_file['chr'][-region_chr_np.shape[0]:] = region_chr_np
        
        h5_file['position'].resize((h5_file['position'].shape[0] + region_pos_np.shape[0]),axis=0)
        h5_file['position'][-region_pos_np.shape[0]:] = region_pos_np

# process all the regions in a enformer tile
def process_region(region_for_chr,chr_info,region_embeddings):
    
    region_for_chr_cpy = region_for_chr.copy() # for updating the list
    pos_embedddings_lst,region_chr,region_pos = [], [], []

    for tt in region_for_chr:
        
        if tt >= (chr_info[1] + enformer_window) and tt <= (chr_info[2] - enformer_window) : # we need region before and after it
            rel_pos = math.floor((tt - chr_info[1]) / enformer_window)
            #print(f"tt {tt}  falls in the current tile - chr_embed_start {chr_info[1]} and chr_embed_end {chr_info[2]} and rel_pos {rel_pos}")
            pos_embedding = region_embeddings[rel_pos - 1 : rel_pos + 2, :].to('cpu')
            assert pos_embedding.shape[0] == 3, "Embeddings should be for 3 consecutive regions" 
            pos_embedding = pos_embedding.mean(axis=0)
            pos_embedddings_lst.append(pos_embedding)
            region_chr.append(int(chr_info[0].replace('chr','').replace('M','23').replace('X','24').replace('Y','25')))
            region_pos.append(tt)
    
    return pos_embedddings_lst,region_chr,region_pos

# process edge regions
def get_rem_region_embeddings(rem_regions, chrs, chr_region_edges_idx, h5_filename):

    h5_file = h5py.File(h5_filename, 'a')
    processed = []
    for tt in rem_regions:
        print(f"Processing edge region {tt} on chromosome {chrs}")
        tt_embeddings_lst,tt_chr_lst,tt_lst = [], [], []
        edge_region_found = 0
        n_embeddings = 3

        for r_start_region in chr_region_edges_idx:
            if tt >= r_start_region and tt < r_start_region + enformer_window: # chrosome matches
                edge_region_found = 1
                print(f"Found embedding file {chrs} - {r_start_region}")
                tt_embedding = torch.load(f'{data_dir}{chrs}_{str(r_start_region)}_embeddings.pt')
                tt_embeddings_lst.append(tt_embedding)

                # before and after
                region_before = r_start_region - enformer_window
                tt_embedding = torch.load(f'{data_dir}{chrs}_{str(region_before)}_embeddings.pt')
                tt_embeddings_lst.append(tt_embedding)

                try: # edge case

                    region_after = r_start_region + enformer_window
                    tt_embedding = torch.load(f'{data_dir}{chrs}_{str(region_after)}_embeddings.pt')

                    tt_embeddings_lst.append(tt_embedding)
                except:

                    print(f"WARNING: for {chrs}:{tt}, there does not exist an embedding to the right of this location at end of chromosome")
                    n_embeddings = 2

                tt_chr_lst.append(int(chrs.replace('chr','').replace('M','23').replace('X','24').replace('Y','25')))
                tt_lst.append(int(tt))
            
                # check if we got all 3 region embeddings
                assert len(tt_embeddings_lst) == n_embeddings, "Embeddings should be for {n_embeddings} regions"

                embeddings_np = np.array(torch.stack(tt_embeddings_lst,axis=0).squeeze(1).mean(axis=0)).reshape(1,-1)
                region_chr_np = np.array(tt_chr_lst).reshape(-1,1)
                region_pos_np = np.array(tt_lst).reshape(-1,1)
                save_hdf5(embeddings_np,region_chr_np,region_pos_np,h5_file)  
                processed.append(tt)
                print(f"Processed {tt}")
                break

        if edge_region_found == 0:
            print(f'Edge regions {tt} on chromosome {chrs} not found in the embedding files')
            break

    h5_file.close()
    # all finished for that chromosome
    assert len(processed) == len(rem_regions), "Some regions still not processed"
    
def get_tiles_for_chr(chrs,region_bed_file):
    
    # load the bed file data
    chr_regions = []
    with open(region_bed_file)as f:
        for line in f:
            if line.strip().split()[0] == chrs:
                chr_regions.append(line.strip().split())
    
    return chr_regions

def convert_bed_to_dict(bed_file):
     # process train and test regions
    regions_dict = defaultdict(list)
    f =  open(bed_file,"r")

    for l in f:
        line = l.split('\t')
        regions_dict[line[0]].append(int(line[1]) + 1) # + 1 as BED file is 0 indexed.
    
    return regions_dict

if __name__ == "__main__":
    
    # reference bed file and fasta file
    region_bed_file = 'data/hg38.196608-1X.windows.fixed1.bed'
    genome_file     = 'data/genome.fa'
    test_bed_file   = 'data/PRECOMP-MID-AVG_GENOME_genome_rois_test.DNACV5_12kb.txt'
    train_bed_file  = 'data/PRECOMP-MID-AVG_GENOME_genome_rois_train.DNACV5_12kb.txt'

    # convert bed to dict
    test_regions  = convert_bed_to_dict(test_bed_file)
    train_regions = convert_bed_to_dict(train_bed_file)
    print(f"Train and test bed file processed")

    # get embedddings for the whole genome one chromosome at a time
    for chrs in ['chr21','chrM','chr1', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 
             'chr2', 'chr20', 'chr21', 'chr22', 
             'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chrX', 'chrY']:
        
        print(f"Processing chromosome {chrs}")
        train_regions_chr = train_regions[chrs]
        test_regions_chr  = test_regions[chrs]
        print(f'Total train regions {len(train_regions_chr)} and test regions {len(test_regions_chr)} to process')
        chr_regions = get_tiles_for_chr(chrs,region_bed_file)
        print(f"Finished getting {len(chr_regions)} tiles for the chromosome")

        train_rem_regions, test_rem_regions,region_edges_idx = get_genome_embeddings(chr_regions,genome_file,train_regions_chr,test_regions_chr,chrs)
        print(f"Genome processing done and the remaining train / test regions are {len(train_rem_regions)}/{len(test_rem_regions)}")

        print(f"region_edges_idx {region_edges_idx}")
        with open("data/region_edges_idx.pickle", "wb") as output_file:
            pickle.dump(region_edges_idx , output_file)
        
        # edge regions embeddings
        if len(train_rem_regions) != 0:
            get_rem_region_embeddings(train_rem_regions,chrs,region_edges_idx,f'data/{chrs}_train_regions_enformer.h5')
            print(f"Completed processing for remaining train regions")
        if len(test_rem_regions) != 0:
            get_rem_region_embeddings(test_rem_regions,chrs,region_edges_idx,f'data/{chrs}_test_regions_enformer.h5')
            print(f"Completed processing for remaining test regions")

        #remove all the edge embeddings on disk
        fileList = glob.glob('data/*.pt', recursive=True) 
        for filePath in fileList:
            os.remove(filePath)