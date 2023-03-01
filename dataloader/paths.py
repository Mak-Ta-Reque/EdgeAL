from dataloader import dataset_base
from dataloader import custom_transforms
import constants

class PathsDataset(dataset_base.DatasetBase):

    def __init__(self, lmdb_handle, base_size, paths):
        super(PathsDataset, self).__init__(lmdb_handle, base_size)
        self.image_path_subset = paths
        self.base_size = base_size

    def __getitem__(self, index):

        image_path = self.image_path_subset[index]
        loaded_npy = self.lmdb_handle.get_numpy_object(image_path)
        image = loaded_npy[:, :, 0:constants.IN_CHANNELS]
        target = loaded_npy[:, :, -1]

        ret_dict = custom_transforms.transform_validation_sample(image, target, base_size=self.base_size)
        return ret_dict
