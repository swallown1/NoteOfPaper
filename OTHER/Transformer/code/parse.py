import argparse

def parse_args():
    arg = argparse.ArgumentParser()
    arg.add_argument('--data_path', default="./data/dataset.pkl", help="the path of dataset")
    arg.add_argument('--min_cnt',type=int,default=20, help="words whose occurred less than min_cnt are encoded as <UNK>.")
    arg.add_argument('--hidden_units', default=512, help="the path of dataset")
    arg.add_argument('--num_blocks', default=6, help="number of encoder/decoder blocks")
    arg.add_argument('--num_epochs', default=20, help="the epochs of train proccess")
    arg.add_argument('--num_heads', default="8", help="the number of mutilpe attention")
    arg.add_argument('--dropout_rate', default="0.1", help="the path of dataset")
    arg.add_argument('--maxlen', type=int,default=10, help="Maximum number of words in a sentence. alias = T.")

    arg.add_argument('--source_train', default="data/train.tags.de-en.de", help="Maximum .")
    arg.add_argument('--target_train', default="data/train.tags.de-en.en", help="Maximum as = T.")
    arg.add_argument('--source_test', default="data/IWSLT16.TED.tst2014.de-en.de.xml", help="Maximum = T.")
    arg.add_argument('--target_test', default="data/IWSLT16.TED.tst2014.de-en.en.xml", help="Maximum as = T.")

    arg.add_argument('--batch_size',type=int, default=32, help="Maximum .")
    arg.add_argument('--lr',type=float, default=0.0001, help="Maximum as = T.")

    return arg.parse_args(args=[])