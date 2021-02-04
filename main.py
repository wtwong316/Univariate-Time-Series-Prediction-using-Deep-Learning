import os
import random
import argparse
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch

from models import DNN, RNN, LSTM, GRU, RecursiveLSTM, AttentionLSTM, CNN
from utils import make_dirs, load_data, plot_full, data_loader, get_lr_scheduler
from utils import mean_percentage_error, mean_absolute_percentage_error, plot_pred_test

# Reproducibility #
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device Configuration #
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main(config):

    # Fix Seed #
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    # Weights and Plots Path #
    paths = [config.weights_path, config.plots_path]

    for path in paths:
        make_dirs(path)

    # Prepare Data #
    data = load_data(config.which_data)[[config.feature]]
    data = data.copy()

    # Plot Time-Series Data #
    if config.plot_full:
        plot_full(config.plots_path, data, config.feature)

    scaler = MinMaxScaler()
    data[config.feature] = scaler.fit_transform(data)

    train_loader, val_loader, test_loader = \
        data_loader(data, config.seq_length, config.train_split, config.test_split, config.batch_size)

    # Lists #
    train_losses, val_losses = list(), list()
    val_maes, val_mses, val_rmses, val_mapes, val_mpes, val_r2s = list(), list(), list(), list(), list(), list()
    test_maes, test_mses, test_rmses, test_mapes, test_mpes, test_r2s = list(), list(), list(), list(), list(), list()

    # Constants #
    best_val_loss = 100
    best_val_improv = 0

    # Prepare Network #
    if config.network == 'dnn':
        model = DNN(config.seq_length, config.hidden_size, config.output_size).to(device)
    elif config.network == 'cnn':
        model = CNN(config.seq_length, config.batch_size).to(device)
    elif config.network == 'rnn':
        model = RNN(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'lstm':
        model = LSTM(config.input_size, config.hidden_size, config.num_layers, config.output_size, config.bidirectional).to(device)
    elif config.network == 'gru':
        model = GRU(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'recursive':
        model = RecursiveLSTM(config.input_size, config.hidden_size, config.num_layers, config.output_size).to(device)
    elif config.network == 'attention':
        model = AttentionLSTM(config.input_size, config.key, config.query, config.value, config.hidden_size, config.num_layers, config.output_size, config.bidirectional).to(device)
    else:
        raise NotImplementedError

    # Loss Function #
    criterion = torch.nn.MSELoss()

    # Optimizer #
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    optim_scheduler = get_lr_scheduler(config.lr_scheduler, optim)

    # Train and Validation #
    if config.mode == 'train':

        # Train #
        print("Training {} started with total epoch of {}.".format(model.__class__.__name__, config.num_epochs))

        for epoch in range(config.num_epochs):
            for i, (data, label) in enumerate(train_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred = model(data)

                # Calculate Loss #
                train_loss = criterion(pred, label)

                # Initialize Optimizer, Back Propagation and Update #
                optim.zero_grad()
                train_loss.backward()
                optim.step()

                # Add item to Lists #
                train_losses.append(train_loss.item())

            # Print Statistics #
            if (epoch+1) % config.print_every == 0:
                print("Epoch [{}/{}]".format(epoch+1, config.num_epochs))
                print("Train Loss {:.4f}".format(np.average(train_losses)))

            # Learning Rate Scheduler #
            optim_scheduler.step()

            # Validation #
            with torch.no_grad():
                for i, (data, label) in enumerate(val_loader):

                    # Prepare Data #
                    data = data.to(device, dtype=torch.float32)
                    label = label.to(device, dtype=torch.float32)

                    # Forward Data #
                    pred_val = model(data)

                    # Calculate Loss #
                    val_loss = criterion(pred_val, label)
                    val_mae = mean_absolute_error(label.cpu(), pred_val.cpu())
                    val_mse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=True)
                    val_rmse = mean_squared_error(label.cpu(), pred_val.cpu(), squared=False)
                    val_mpe = mean_percentage_error(label.cpu(), pred_val.cpu())
                    val_mape = mean_absolute_percentage_error(label.cpu(), pred_val.cpu())
                    val_r2 = r2_score(label.cpu(), pred_val.cpu())

                    # Add item to Lists #
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
                    val_mses.append(val_mse.item())
                    val_rmses.append(val_rmse.item())
                    val_mpes.append(val_mpe.item())
                    val_mapes.append(val_mape.item())
                    val_r2s.append(val_r2.item())

            if (epoch + 1) % config.print_every == 0:

                # Print Statistics #
                print("Val Loss {:.4f}".format(np.average(val_losses)))
                print("Val  MAE : {:.4f}".format(np.average(val_maes)))
                print("Val  MSE : {:.4f}".format(np.average(val_mses)))
                print("Val RMSE : {:.4f}".format(np.average(val_rmses)))
                print("Val  MPE : {:.4f}".format(np.average(val_mpes)))
                print("Val MAPE : {:.4f}".format(np.average(val_mapes)))
                print("Val  R^2 : {:.4f}".format(np.average(val_r2s)))

                # Save the model Only if validation loss decreased #
                curr_val_loss = np.average(val_losses)

                if curr_val_loss < best_val_loss:
                    best_val_loss = min(curr_val_loss, best_val_loss)
                    torch.save(model.state_dict(), os.path.join(config.weights_path, 'BEST_{}.pkl'.format(model.__class__.__name__)))

                    print("Best model is saved!\n")
                    best_val_improv = 0

                elif curr_val_loss >= best_val_loss:
                    best_val_improv += 1
                    print("Best Validation has not improved for {} epochs.\n".format(best_val_improv))

    elif config.mode == 'test':

        # Load the Model Weight #
        model.load_state_dict(torch.load(os.path.join(config.weights_path, 'BEST_{}.pkl'.format(model.__class__.__name__))))

        # Test #
        with torch.no_grad():
            for i, (data, label) in enumerate(test_loader):

                # Prepare Data #
                data = data.to(device, dtype=torch.float32)
                label = label.to(device, dtype=torch.float32)

                # Forward Data #
                pred_test = model(data)

                # Convert to Original Value Range #
                pred_test = pred_test.data.cpu().numpy()
                label = label.data.cpu().numpy().reshape(-1, 1)

                pred_test = scaler.inverse_transform(pred_test)
                label = scaler.inverse_transform(label)

                # Calculate Loss #
                test_mae = mean_absolute_error(label, pred_test)
                test_mse = mean_squared_error(label, pred_test, squared=True)
                test_rmse = mean_squared_error(label, pred_test, squared=False)
                test_mpe = mean_percentage_error(label, pred_test)
                test_mape = mean_absolute_percentage_error(label, pred_test)
                test_r2 = r2_score(label, pred_test)

                # Add item to Lists #
                test_maes.append(test_mae.item())
                test_mses.append(test_mse.item())
                test_rmses.append(test_rmse.item())
                test_mpes.append(test_mpe.item())
                test_mapes.append(test_mape.item())
                test_r2s.append(test_r2.item())

            # Print Statistics #
            print("Test {}".format(model.__class__.__name__))
            print("Test  MAE : {:.4f}".format(np.average(test_maes)))
            print("Test  MSE : {:.4f}".format(np.average(test_mses)))
            print("Test RMSE : {:.4f}".format(np.average(test_rmses)))
            print("Test  MPE : {:.4f}".format(np.average(test_mpes)))
            print("Test MAPE : {:.4f}".format(np.average(test_mapes)))
            print("Test  R^2 : {:.4f}".format(np.average(test_r2s)))

            # Plot Figure #
            plot_pred_test(pred_test, label, config.plots_path, config.feature, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')
    parser.add_argument('--feature', type=str, default='Appliances', help='extract which feature for prediction')

    parser.add_argument('--seq_length', type=int, default=5, help='window size')
    parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size')

    parser.add_argument('--network', type=str, default='dnn',
                        choices=['dnn', 'cnn', 'rnn', 'lstm', 'gru', 'recursive', 'attention'])

    parser.add_argument('--input_size', type=int, default=1, help='input_size')
    parser.add_argument('--hidden_size', type=int, default=10, help='hidden_size')
    parser.add_argument('--num_layers', type=int, default=1, help='num_layers')
    parser.add_argument('--output_size', type=int, default=1, help='output_size')
    parser.add_argument('--bidirectional', type=bool, default=False, help='use bidirectional or not')

    parser.add_argument('--key', type=int, default=8, help='key')
    parser.add_argument('--query', type=int, default=8, help='query')
    parser.add_argument('--value', type=int, default=8, help='value')

    parser.add_argument('--which_data', type=str, default='./data/energydata_complete.csv', help='which data to use')
    parser.add_argument('--weights_path', type=str, default='./results/weights/', help='weights path')
    parser.add_argument('--plots_path', type=str, default='./results/plots/', help='plots path')

    parser.add_argument('--train_split', type=float, default=0.8, help='train_split')
    parser.add_argument('--test_split', type=float, default=0.5, help='test_split')

    parser.add_argument('--num_epochs', type=int, default=200, help='total epoch')
    parser.add_argument('--print_every', type=int, default=10, help='print statistics for every default epoch')

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', help='learning rate scheduler', choices=['step', 'plateau', 'cosine'])

    parser.add_argument('--plot_full', type=bool, default=False, help='plot full graph or not')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    config = parser.parse_args()

    torch.cuda.empty_cache()
    main(config)