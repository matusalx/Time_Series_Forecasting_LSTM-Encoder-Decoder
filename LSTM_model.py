import os
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class LSTMDataset(Dataset):
    def __init__(self, x, y):
        self.x_train = x
        self.y_train = y

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x, y = torch.from_numpy(self.x_train[idx].values), torch.from_numpy(self.y_train[idx].values)
        x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
        return x, y




class LSTM(nn.Module):
    def __init__(self, num_classes=20, input_size=6, hidden_size=64, num_layers=1):
        super(LSTM, self).__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))

        # Propagate input through LSTM
        all_out, (h_out, _) = self.lstm(x, (h_0, c_0))

        h_out = h_out.view(-1, self.hidden_size)

        out = self.fc(h_out)
        out = out.unsqueeze(1)
        out = torch.sigmoid(out)*2

        return out




def lstm_train(x_train, y_train, x_val, y_val):

    num_epochs = 20
    learning_rate = 0.0001
    input_size = x_train[0].shape[1]
    hidden_size = 64
    num_layers = 1
    num_classes = 20
    batch_size = 1
    min_val_loss = torch.inf

    train_data = LSTMDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

    val_data = LSTMDataset(x_val, y_val)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Create directory for saving models
    model_dir = os.path.join(os.getcwd(), 'Models')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    lstm_dir = os.path.join(model_dir, 'LSTM.pt')

    model = LSTM(num_classes, input_size, hidden_size, num_layers)

    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
    all_epoch_val_lost = []
    all_epoch_train_lost = []

    # Train the model
    for epoch in range(num_epochs):
        train_loss = 0
        val_loss = 0
        model.train()
        for x_, y_ in train_loader:
            outputs = model(x_)
            optimizer.zero_grad()

            # outputs = outputs.unsqueeze(2)
            # y_ = y_.squeeze(2)

            loss = criterion(outputs, y_)

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
                    outputs = model(x_)

                    # Calculate loss
                    loss = criterion(outputs, y_)
                    val_loss += loss

                epoch_val_loss = val_loss / len(val_loader)
                all_epoch_val_lost.append(epoch_val_loss)
                print("epoch validation Loss: {:.6f}".format(epoch_val_loss))
                # Average validation loss
                epoch_val_loss = val_loss / len(val_loader)
                # If the validation loss is at a minimum
                if val_loss < min_val_loss:
                    # Save the model
                    torch.save(model.state_dict(), lstm_dir)
                    print('Saving model to {}'.format(lstm_dir))
                    min_val_loss = val_loss
                    model_to_return = model

    plt.title('LSTM')
    plt.style.use('fivethirtyeight')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(all_epoch_train_lost)
    plt.plot(all_epoch_val_lost)
    plt.legend(['train loss', 'validation loss'])
    plt.show()

    return model_to_return


