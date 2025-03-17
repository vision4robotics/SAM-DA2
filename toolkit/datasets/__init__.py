from .UAVDark135 import UAVDark135Dataset
from .nut_l import NUT_LDataset

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):
        """
        Args:
            name: dataset name 'UAVDark70', 'UAV', 'NAT', 'NAT'
            dataset_root: dataset root
            load_img: wether to load image
        Return:
            dataset
        """
        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAVDark135' == name:
            dataset = UAVDark135Dataset(**kwargs)
        elif 'NUT_L' == name:
            dataset = NUT_LDataset(**kwargs)
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset

