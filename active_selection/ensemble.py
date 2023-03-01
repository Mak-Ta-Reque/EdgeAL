import random
from active_selection import ceal, vote_entropy, softmax_entropy, random_selection, regional_vote_entropy, regional_view_entropy_kl
import numpy as np
import os
from dataloader.dataset_base import OverlapHandler
import constants
from dataloader.indoor_scenes import get_active_dataset
class EnsembleSelector:
    def __init__(self, dataset, lmdb_handle, base_size, batch_size, train_set, args):
        self.lmdb_handle = lmdb_handle
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.args = args
        self.train_set = train_set
        

    def select_ceal(self, model, training_set):
        start_entropy_threshold = 0.0275
        entropy_change_per_selection =  0.001815

        selector =  ceal.CEALSelector( self.dataset, self.lmdb_handle, self.base_size, self.batch_size, self.train_set.num_classes, start_entropy_threshold, entropy_change_per_selection)

        selected_list = selector.ranking(model, training_set)
        return selected_list

    def select_mcdr(self, model, training_set):
        mcdr_soft = vote_entropy.VoteEntropySelector(self.dataset, self.lmdb_handle, self.base_size, self.batch_size, self.train_set.num_classes, True)
        selected_list = mcdr_soft.rankings(model, training_set)
        return selected_list

    def select_mcdr_region(self, model, training_set):
        train_set = get_active_dataset("vote_entropy_region")(self.args.dataset, self.lmdb_handle, self.args.superpixel_dir, self.args.base_size, 'seedset_0')
    
        if not self.args.no_overlap:
            overlap_handler = OverlapHandler(os.path.join(constants.SSD_DATASET_ROOT, self.args.dataset, "raw", "selections", 'coverage_superpixel'), 1, memory_hog_mode=True)
        selector = regional_vote_entropy.RegionalVoteEntropySelector(self.args.dataset, self.lmdb_handle, "superpixel", self.args.base_size, self.args.batch_size, self. train_set.num_classes, 100, overlap_handler, mode="superpixel")
        selected_list = selector.ranksings(model, train_set)
        return selected_list

    def select_softmax_entropy(self, model, training_set):
        sofmax_entropy = softmax_entropy.SoftmaxEntropySelector(self.dataset, self.lmdb_handle, self.base_size, self.batch_size, self.train_set.num_classes)
        selected_list = sofmax_entropy.ranking(model, training_set)
        return selected_list
    
    def select_view_entropy(self,  model, training_set):
        overlap_handler = None
        train_set = get_active_dataset("viewmc_kldiv_region")(self.args.dataset, self.lmdb_handle, self.args.superpixel_dir, self.args.base_size, 'seedset_0')
        if not self.args.no_overlap:
            overlap_handler = OverlapHandler(os.path.join(constants.SSD_DATASET_ROOT, self.args.dataset, "raw", "selections", 'coverage_superpixel'), 1, memory_hog_mode=True)
        selector = regional_view_entropy_kl.RegionalViewEntropyWithKldivSelector(self.args.dataset, self.lmdb_handle, "superpixel", self.args.base_size, self.train_set.num_classes, self.args.region_size, overlap_handler, mode="superpixel")
        return selector.rankings(model, train_set)


    def select_random(self, model, training_set):
        random_select = random_selection.RandomSelector()
        return list(random_select.ranking(model, training_set))
    def select_softentropy():
        selected_list = []
        return selected_list
    def new_ranking(self, list_student):
        studens = list_student[0]
        scores_dict = {}
        for st in studens:
            scores = []
            for course in list_student:
                #if st in course:
                scores.append(course.index(st))
                #else:
                #    scores.append(len(course)//2)

            scores_dict[st] = np.sum([s*s for s in scores])
        overall_ransks = sorted(scores_dict, key=lambda x: scores_dict[x])
        return overall_ransks




    def intersection(self, lst1, lst2, lst3):
        return list(set(lst1).intersection(lst2).intersection(lst3))

    def select_next_batch(self, model, training_set, selection_count):
        if not len(training_set.remaining_image_paths)  == 0:
            ceal_selction = self.select_ceal(model, training_set)
            mcdr_selction = self.select_mcdr(model, training_set)
            softmax_entroy_selection = self.select_softmax_entropy(model, training_set)
            mcdr_region_selection = self.select_mcdr_region(model, training_set)
            view_entroy_selections = self.select_view_entropy(model, training_set)
            #random_selection = self.select_random(model, training_set)[:selection_count]
            #print( random_selection)
            inter_selection = self.new_ranking([ceal_selction, mcdr_selction, softmax_entroy_selection, mcdr_region_selection, view_entroy_selections]) 
            
            selected_samples = inter_selection[:selection_count]
            training_set.expand_training_set(selected_samples)
        else:
            training_set.expand_training_set([])
