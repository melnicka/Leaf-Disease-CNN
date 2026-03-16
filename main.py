from src.utils import load_config, parse_args
from src.builder import train_model, make_predictions

if __name__ == '__main__':
    args = parse_args()

    if args.command == 'train':
        conf_list = [f"{args.root_dir}/{conf}" for conf in args.config]
        cfg = load_config(conf_list)
        train_model(args.name, cfg)

    elif args.command == 'predict':
        preds = make_predictions(
                args.name,
                args.input_data,
                args.base_dir
        )

        if args.save:
            with open(args.save, "w") as f:
                f.write(str(preds))
        
        print(preds)



        

 
