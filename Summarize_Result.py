from Run_Aux_Training import *
import glob


task_dir = "x_scale"



summary_dir = "Summaries"
if not os.path.exists(summary_dir):
    os.mkdir(summary_dir)

checkpoint_dict = {"grid5": "grid5-1.0-0.01_MLP-5-128", "grid25": None, "circle2": None, "circle2c": None, "random9-3_1": None}


dataset_name = "grid5"
checkpoint_pattern = os.path.join(aux_model_folder, checkpoint_dict[dataset_name], "*.pth")
checkpoint_path = glob.glob(checkpoint_pattern)[0]
print(checkpoint_path)

checkpoint = torch.load(checkpoint_path)
print(checkpoint)

args = checkpoint["args"]
rng = np.random.RandomState(seed=args.seed)
dataset = Synthetic_Dataset(args.data, rng, std=args.mog_std, scale=args.mog_scale, sample_per_mode=1000, with_neg_samples=True)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

aux_classifier_loaded = MLP(n_hidden_layers=args.layers, n_hidden_neurons=args.hidden, input_dim=2, output_dim=dataset.n + 1, type="D", use_bias=True).to(device)


exp_data_list = glob.glob(os.path.join(summary_dir, task_dir, f"{dataset_name}*.pth"))





