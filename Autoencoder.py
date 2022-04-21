import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Data##################################
class AutoencoderDataset(Dataset):

    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        x1 = torch.from_numpy(self.df.iloc[idx].values)
        y1 = torch.from_numpy(self.df.iloc[idx].values)

        x1, y1 = x1.type(torch.FloatTensor), y1.type(torch.FloatTensor),
        return x1, y1

# Model###############################
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()
          
        # Building an linear encoder with Linear
        # layer followed by Relu activation function
        # 20 ==> 5
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(19, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 5),
        )
          
        # Building an linear decoder with Linear
        # layer followed by Relu activation function
        # The Sigmoid activation function
        # outputs the value between 0 and 1
        # 5 ==> 20
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(5, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 19),
            # torch.nn.Sigmoid()
            torch.nn.Tanh()                
        )
  
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded


def train_encoder(df_train, df_test):

    train_data = AutoencoderDataset(df_train)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    test_data = AutoencoderDataset(df_test)
    test_loader = DataLoader(test_data, batch_size=100, shuffle=True)

    model = AE()
    
    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-8)

    epochs = 20
    min_train_loss = torch.inf
    # Create directory for saving models
    model_dir = os.path.join(os.getcwd(), 'Models')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    ae_dir = os.path.join(model_dir, 'AutoEncoder.pt')

    all_epoch_test_lost = []
    all_epoch_train_lost = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        test_loss = 0
        model.train()

        for (data, _) in train_loader:
            # Output of Autoencoder
            reconstructed, _ = model(data)

            optimizer.zero_grad()
            loss = loss_function(reconstructed, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_train_loss = train_loss / len(train_loader)
        print("epoch training Loss: {:.6f}".format(epoch_train_loss))
        all_epoch_train_lost.append(epoch_train_loss)
        # Test data
        if epoch % 1 == 0:
            with torch.no_grad():
                model.eval()
                for (data, _) in test_loader:
                    # Generate predictions
                    reconstructed, _ = model(data)

                    # Calculate loss
                    loss = loss_function(reconstructed, data)
                    test_loss += loss

                epoch_test_loss = test_loss / len(test_loader)
                print("epoch validation Loss: {:.6f}".format(epoch_test_loss))
                # If the validation loss is at a minimum
                if test_loss < min_train_loss:
                    # Save the model
                    torch.save(model.state_dict(), ae_dir)
                    print('Saving model to {}'.format(ae_dir))
                    min_train_loss = epoch_test_loss
                    model_to_return = model
        all_epoch_test_lost.append(test_loss)

    plt.title('AutoEncoder')
    plt.style.use('fivethirtyeight')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(all_epoch_train_lost)
    plt.plot(all_epoch_test_lost)
    plt.legend(['train_loss', 'test_loss'])
    plt.show()

    return model_to_return
  



