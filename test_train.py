import os
import create
import yaml
import argparse

cur_dir = os.path.abspath(__file__)
cur_dir = os.path.dirname(cur_dir)

def train(model_config_path):
    with open(model_config_path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    ## Preprocessing 
    # update: input_length, inp_dim, output_dim, embdding_dim, embedding_matrix

    model_gen = create.ModelCreator(**model_config)
    conf, model = model_gen.build()
    print(conf)
    model.summary()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-mp',
        '--model_path',
        help='model_path',
    )

    args = parser.parse_args()

    train(args.model_path)