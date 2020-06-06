
"""
Adapted from: https://github.com/matterport/Mask_RCNN
------------------------------------------------------
"""

import mute
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from config import Config
import model as modellib
import utils as utils


parser = argparse.ArgumentParser(description='MAGnet: Implementation in Keras')
parser.add_argument('--data_path', dest='data_path', default='/media/dataHD3/MAGnet/data/', help='Database directory')
parser.add_argument('--log_path', dest='log_path', default='/media/dataHD3/MAGnet/logs/', help='Logs and checkpoints directory')
parser.add_argument('--dataset', dest='dataset', default='flickr30k', help='Choose dataset (flickr30, refclef, visual_gnome)')
parser.add_argument('--load_mode', dest='load_mode', default='imagenet', help='Specify checkpoint model .h5 (default imagenet to retrain)')
args_dict = parser.parse_args(args=[])


############################################################
#  Configurations
############################################################
class CustomConfig(Config):
    """Custom configuration for each dataset.
    """
    if args_dict.dataset == "flickr30k":
        NAME = "flickr30k"
        JSON = "flickr30k.json"
        VOCAB_SIZE = 4086
        SEQ_LEN = 18

    elif args_dict.dataset == "refclef":
        NAME = "refclef"
        JSON = "refclef.json"
        VOCAB_SIZE = 1856
        SEQ_LEN = 18

    elif args_dict.dataset == "visual_genome":
        NAME = "visual_genome"
        JSON = "visual_genome.json"
        VOCAB_SIZE = 18481
        SEQ_LEN = 18


############################################################
#  Dataset
############################################################
class CustomDataset(utils.Dataset):
    """Custom loading data for each dataset.
    """
    def __init__(self):
        utils.Dataset.__init__(self, base_image_path=args_dict.data_path)

    def load_data(self, config, subset):
        """Load a subset of the dataset.
        subset: What to load (train, val, test)
        """

        self.data_folder = args_dict.data_path
        self.json_file = config.JSON
        self.data = json.load(open(os.path.join(self.data_folder, self.json_file), 'r'))
        self.list_IDs = len(self.data)
        self.shuffle = True
        self.imgs = self.data['images']

        # Split image and vocabs
        splits, image_ids, self.vocab2idx, self.idx2vocab = self.get_splits_and_vocab()
        sample_list = splits[subset]

        self.list_IDs = sample_list
        self.image_ids1 = image_ids[subset]
        self.vocab_size = len(self.vocab2idx.keys())
        self.glove_path = os.path.join(self.data_folder, 'glove.42B.300d.txt')
        self.glove_dim = 300


        # Add images along with captions
        for i in tqdm(range(len(self.imgs))):
            captions = self.imgs[i]['captions']
            encoded_captions = self.imgs[i]['captions_encoded']
            tmp_boxes = self.imgs[i]['bboxes']
            boxes = []
            for j in range(len(captions)):
                boxes.append(np.array([tmp_boxes[j]['y1'], tmp_boxes[j]['x1'], tmp_boxes[j]['y2'], tmp_boxes[j]['x2']]))

            boxes = np.array(boxes)
            self.add_image(
                config.NAME, image_id=i,
                path=self.imgs[i]['filepath'],
                width=self.imgs[i]["width"],
                height=self.imgs[i]["height"],
                captions=captions,
                encoded_captions=encoded_captions,
                bboxes=boxes)


    def get_embmat(self):
        _START_VOCAB = ["<start>", "UNK", "<eos>"]

        # The vocab size of the corpus we've downloaded
        vocab_size = int(1.9e6)
        emb_matrix = np.zeros([self.vocab_size + 1, self.glove_dim])

        idx = 0
        random_init = True
        # randomly initialize the special tokens
        if random_init:
            for word in _START_VOCAB:
                emb_matrix[int(self.vocab2idx[word]), :] = np.random.randn(1, self.glove_dim)
                idx += 1

        noemb = np.zeros(self.vocab_size + 1)

        # go through glove vectors
        with open(self.glove_path, 'r') as fh:
            for line in tqdm(fh, total=vocab_size):
                line = line.lstrip().rstrip().split(" ")
                word = line[0]

                if word in self.vocab2idx:

                    vector = list(map(float, line[1:]))
                    if self.glove_dim != len(vector):
                        raise Exception("You set --glove_path=%s but --embedding_size=%i. If you set --glove_path yourself then make sure that --embedding_size matches!" % (self.glove_path, self.glove_dim))
                    emb_matrix[int(self.vocab2idx[word]), :] = vector
                    noemb[self.vocab2idx[word]] = 1

        for i in range(self.vocab_size):
            if noemb[i + 1] == 0:
                emb_matrix[i + 1, :] = emb_matrix[int(self.vocab2idx[_START_VOCAB[1]]), :]
                idx += 1

        return emb_matrix


    def get_splits_and_vocab(self):
        splits = {'train': [], 'val': [], 'test': []}
        idxs = {'train': [], 'val': [], 'test': []}

        for i, img in enumerate(self.data['images']):
            splits[img['split']].append(i)
            idxs[img['split']].append(i)

        vocab = self.data['idx_to_word']
        vocab_int_inv = {}
        for idx in vocab.keys():
            vocab_int_inv[vocab[idx]] = int(idx)
        return splits, idxs, vocab_int_inv, vocab


    def load_gtbox(self, image_id):
        gtbox = self.image_info[image_id]['bboxes']
        captions = self.image_info[image_id]['captions']
        en_captions = self.image_info[image_id]['encoded_captions']
        return gtbox, captions, en_captions


############################################################
#  Training
############################################################
if __name__ == '__main__':
    config = CustomConfig()
    config.display()

    # Training dataset
    dataset_train = CustomDataset()
    dataset_train.load_data(config, "train")

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_data(config, "val")

    # Get embedded vector for caption
    embed_mat = dataset_train.get_embmat()

    # Initialize Model
    model = modellib.MAGnet(mode="training", config=config, embed_mat=embed_mat, model_dir=args_dict.log_path)
    load_mode = args_dict.load_mode

    # Load weights
    if load_mode.lower() == "imagenet":
        print("loading imagenet weights")
        model_path = model.get_imagenet_weights()
    else:
        model_path = load_mode
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training
    print("Training network heads")
    config.LOSS_WEIGHTS = {
        "lang_language_loss": 1.,
        "rpn_class_loss": 1.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1.,
        "mrcnn_bbox_loss": 1.
    }
    model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1200, layers='heads')
