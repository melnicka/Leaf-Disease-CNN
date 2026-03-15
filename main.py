from src.utils import load_config, parse_args
from src.builder import train_model, make_predictions

if __name__ == '__main__':
    args = parse_args()

    if args.command == 'train':
        cfg = load_config(args.config)
        train_model(args.name, cfg)

    elif args.command == 'predict':
        preds = make_predictions(
                args.model_name,
                args.input_data,
                args.root_dir
        )
        print(preds)

    if args.save:
        with open(args.save, "w") as f:
            f.write(str(preds))


        


