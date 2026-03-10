from src.utils import load_config, parse_args
from src.builder import train_model
from src.model import LeafCNN
import torch

# testing
if __name__ == '__main__':
    args = parse_args()

    if args.command == 'train':
        cfg = load_config(args.config)
        train_model(args.name, cfg)

    elif args.command == 'predict':
        model = LeafCNN
        state_dict = torch.load(args.model)
        model.load_state_dict(state_dict)

        # TODO: implemet predicting from the user input
        


