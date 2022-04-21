import os
import random
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Resources
# https://www.projectpro.io/recipes/create-seq2seq-modelling-models-pytorch-also-explain-encoder-and-decoder
# https://github.com/lkulowski/LSTM_encoder_decoder/blob/master/code/lstm_encoder_decoder.py
# https://www.kaggle.com/code/omershect/learning-pytorch-seq2seq-with-m5-data-set/notebook#Building-the-Seq2Seq-Model-
# https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb


class LSTMEncoderDecoderDataset(Dataset):
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x, y = torch.from_numpy(self.x_train[idx].values), torch.from_numpy(self.y_train[idx].values)
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
        return x, y


class Encoder(nn.Module):
    def __init__(self, n_features, hidden_dim=64):
        super(Encoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.rnn1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.35
        )

    def forward(self, x):

        h_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))

        c_1 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_dim))

        outputs, (hidden, cell) = self.rnn1(x, (h_1, c_1))

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size=1, hidden_dim=64, decode_output=1):
        super(Decoder, self).__init__()

        self.seq_len, self.input_size = input_size, input_size
        self.hidden_dim, self.decode_output = hidden_dim, decode_output

        self.rnn1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.output_layer = nn.Linear(self.hidden_dim, decode_output)

    def forward(self, x, input_hidden, input_cell):
        x = x.reshape((1, 1, 1))

        x, (hidden_n, cell_n) = self.rnn1(x, (input_hidden, input_cell))

        x = self.output_layer(x)
        x = torch.sigmoid(x) * 2
        return x, hidden_n, cell_n


class Seq2Seq(nn.Module):

    def __init__(self, n_features=20, hidden_dim=64, output_length=20, teacher_force_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.teacher_force_ratio = teacher_force_ratio
        self.encoder = Encoder(n_features, hidden_dim)

        self.output_length = output_length

        self.decoder = Decoder()

    def forward(self, x, prev_y, target_y):
        encoder_output, hidden, cell = self.encoder(x)

        # Prepare place holder for decoder output
        targets_ta = []
        # prev_output become the next input to the LSTM cell
        prev_output = prev_y

        # iterate over LSTM - according to the required output days
        for out_days in range(self.output_length):
            decode_output, prev_hidden, prev_cell = self.decoder(prev_output, hidden, cell)
            hidden, cell = prev_hidden, prev_cell

            prev_output = target_y[:, out_days:out_days + 1, :] if random.random() < self.teacher_force_ratio \
                else decode_output

            targets_ta.append(prev_output.reshape(1))

        targets = torch.stack(targets_ta)

        return targets


def lstm_encoder_decoder(x_train, y_train, x_val, y_val):

    epochs = 20
    rate_learning = 0.01
    batch_size = 1
    n_features = x_train[0].shape[1]
    min_val_loss = torch.inf
    # Create directory for saving models
    model_dir = os.path.join(os.getcwd(), 'Models')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    lstm_encoder_decoder_dir = os.path.join(model_dir, 'LSTM_encoder_decoder.pt')

    train_data = LSTMEncoderDecoderDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    val_data = LSTMEncoderDecoderDataset(x_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = Seq2Seq(n_features=n_features, hidden_dim=64)

    optimizer = torch.optim.Adam(model.parameters(), lr=rate_learning, weight_decay=1e-5)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5e-3, eta_min=1e-8, last_epoch=-1)

    all_epoch_val_lost = []
    all_epoch_train_lost = []

    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for x_, y_ in train_loader:
            optimizer.zero_grad()

            last_x = x_[:, 260 - 1: 260, 0: 1]

            seq_pred = model(x_, last_x, y_)

            loss = criterion(seq_pred, y_)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)
        all_epoch_train_lost.append(epoch_train_loss)
        print("epoch training Loss: {:.6f}".format(epoch_train_loss))

        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                for x_, y_ in val_loader:
                    # Generate predictions
                    last_x = x_[:, 260 - 1: 260, 0: 1]
                    seq_pred = model(x_, last_x, y_)
                    # Calculate loss
                    loss = criterion(seq_pred, y_)
                    val_loss += loss.item()

                epoch_val_loss = val_loss / len(val_loader)
                all_epoch_val_lost.append(epoch_val_loss)
                print("epoch validation Loss: {:.6f}".format(epoch_val_loss))

                # If the validation loss is at a minimum
                if val_loss < min_val_loss:
                    # Save the model
                    torch.save(model.state_dict(), lstm_encoder_decoder_dir)
                    print('Saving model to {}'.format(lstm_encoder_decoder_dir))
                    min_val_loss = val_loss
                    model_to_return = model

    plt.title('LSTM_Encoder_Decoder')
    plt.style.use('fivethirtyeight')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(all_epoch_train_lost)
    plt.plot(all_epoch_val_lost)
    plt.legend(['train_lost', 'validation_lost'])
    plt.show()

    return model_to_return