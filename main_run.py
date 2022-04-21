import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split

import Autoencoder
import LSTM_encode_decoder
import LSTM_model
import importlib



# wdir = os.path.dirname(os.path.realpath(__file__))
wdir = os.getcwd()
sys.path.append(wdir)

df = pd.read_excel(r'Dj.xlsx', sheet_name='Final Data')


columns = ['Date', 'Return', 'SMAVG (50)', 'MACD (12,26)', 'SIG (9)', 'Diff', 
           "William's R", 'RSI (14)',	'UBB(2)', 'BollMA (20)', 'LBB (2)',	'BollW',
           'B%', 'CMCI (13)', '+DMI (14)', '-DMI', 'ADX', '%DS(5)',	'%DSS(3)',
           'TE UB (20,2)', 'TE LB (20,2)']

df = df[columns]
df = df.sort_values(by='Date', ascending=True)
df.drop(['Date'], inplace=True, axis=1)

# standartize and normalize
# define min max scaler
scaler_normalize = StandardScaler()
scaler_minimax = MaxAbsScaler()

# transform data,except for Return
columns2 = ['SMAVG (50)', 'MACD (12,26)', 'SIG (9)', 'Diff', 
            "William's R", 'RSI (14)',	'UBB(2)', 'BollMA (20)', 'LBB (2)',	'BollW',
            'B%', 'CMCI (13)', '+DMI (14)', '-DMI', 'ADX', '%DS(5)',	'%DSS(3)',
            'TE UB (20,2)', 'TE LB (20,2)']

df_scaled = pd.DataFrame(scaler_minimax.fit_transform(df[columns2].values), columns=columns2, index=df.index)
df_scaled = pd.DataFrame(scaler_normalize.fit_transform(df_scaled[columns2].values), columns=columns2, index=df.index)
#  add original Return to data
df = pd.concat([df['Return'], df_scaled], axis=1)


######################################################################################################################
# Autoencoder
# Remove Return variable from Autoencoder
ae_df = df.drop(['Return'], axis=1)

ae_train, ae_test = train_test_split(ae_df, test_size=0.2)

importlib.reload(Autoencoder)
# Training and saving
encoder_model = Autoencoder.train_encoder(ae_train, ae_test)
# Loading saved model
auto_encoder = Autoencoder.AE()
model_dir = os.path.join(wdir, 'Models', 'AutoEncoder.pt')
auto_encoder.load_state_dict(torch.load(model_dir))
auto_encoder.eval()
# Create whole dataframe from autoencoder
encoded_df = []
for x in df.iloc:
    x_torch = torch.Tensor(x.values)
    x_decoded, x_encoded = auto_encoder(x_torch)
    x_encoded = pd.DataFrame(columns=['x1', 'x2', 'x3', 'x4', 'x5'], data=x_encoded.detach().numpy().reshape(1, 5))
    encoded_df.append(x_encoded)

encoded_df = pd.concat(encoded_df)
encoded_df.reset_index(inplace=True, drop=True)
# Add original Return variable
ae_df = pd.concat([df['Return'], encoded_df], axis=1)
# Use df = ae_df if using encoding data, else use df instead of ae_df
df = ae_df

######################################################################################################################
# Prepare data for LSTM models
train_size = 2263
validation_size = len(df) - train_size
test_size = len(df) - train_size

df_train = df[:train_size-validation_size]
df_validation = df[-test_size-validation_size:-test_size]
df_test = df[-test_size:]

train_size = 2263-505
X_train, Y_train = [], []
X_val, Y_val = [], []
X_test, Y_test = [], []
for x in range(0, train_size-260-20, 20):
    X_train.append(df_train.iloc[x:x+260])
    Y_train.append(df_train.iloc[x+260+1:x+260+1+20][['Return']])

for x in range(0, validation_size-260-20, 20):
    X_val.append(df_validation.iloc[x:x+260])
    Y_val.append(df_validation.iloc[x+260+1:x+260+1+20][['Return']])

for x in range(0, test_size-260-20, 20):
    X_test.append(df_test.iloc[x:x+260])
    Y_test.append(df_test.iloc[x+260+1:x+260+1+20][['Return']])

######################################################################################################################
# LSTM_model
importlib.reload(LSTM_model)

lstm = LSTM_model.lstm_train(X_train, Y_train, X_val, Y_val)

######################################################################################################################
# LSTM_encoder_decoder model

importlib.reload(LSTM_encode_decoder)

lstm_enoder_decoder = LSTM_encode_decoder.lstm_encoder_decoder(X_train, Y_train, X_val, Y_val)

######################################################################################################################
# Start forecasting on unseen data

# Load saved model, choose which model to load: LSTM, LSTM_encoder_decoder
load_LSTM_simple = True
#load_LSTM_simple = False
if load_LSTM_simple: model = LSTM_model.LSTM(input_size=X_train[0].shape[1]);model_dir = os.path.join(wdir, 'Models', 'LSTM.pt')
else: model = LSTM_encode_decoder.Seq2Seq(n_features=X_train[0].shape[1]);model_dir = os.path.join(wdir, 'Models', 'LSTM_encoder_decoder.pt')
model.load_state_dict(torch.load(model_dir))
model.eval()
if not load_LSTM_simple:model.teacher_force_ratio = 0
print(model._get_name())


# converts data to tensor, feed to model and plot forecast for each x input
fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(15, 12), sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.5)
fig.suptitle("Forecasting results per input", fontsize=18, y=0.95)
predicted_values = []
for x, ax in zip(range(len(Y_test)), axs.reshape(-1)):
    x_for_model = torch.Tensor(X_test[x].values).unsqueeze(0)
    last_day_x = torch.Tensor(X_test[x]['Return'][-1:].values)
    y_for_model = torch.Tensor(Y_test[x].values).unsqueeze(0)
    if model._get_name() == 'Seq2Seq': y_pred = model(x_for_model, last_day_x, y_for_model)
    else: y_pred = model(x_for_model)
    y_pred = y_pred.reshape(-1).detach().numpy()
    predicted_values.append(y_pred)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Return')
    ax.plot(Y_test[x]['Return'].to_list())
    ax.plot(y_pred)
    ax.legend(['True values', 'Predicted values'])

plt.show()

######################################################################################################################
# The final result is to  see how the portfolio value will increase with model prediction
final_portfolio = []
for y_pred, y_true in zip(predicted_values, Y_test):
    y_pred = y_pred.reshape(-1)
    y_true['Predicted'] = y_pred
    final_portfolio.append(y_true)
final_portfolio = pd.concat(final_portfolio)

final_portfolio['Predicted_gain'] = [1/x['Return'] if x['Predicted'] < 1 else x['Return'] for x in final_portfolio.iloc()]
final_portfolio['Predicted_portfolio'] = final_portfolio['Predicted_gain'].cumprod()
final_portfolio['Index_portfolio'] = final_portfolio['Return'].cumprod()

final_portfolio.plot()
plt.show()











'''
from LSTM_encode_decoder import LSTMEncoderDecoderDataset
test_data = LSTMEncoderDecoderDataset(X_test, Y_test)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
with torch.no_grad():
    lstm_enoder_decoder.eval()
    criterion = torch.nn.MSELoss()
    test_loss = 0
    test_loss_list = []
    predicted_values = []
    for x_, y_ in test_loader:
        # Generate predictions
        last_x = x_[:, 260 - 1: 260, 0: 1]
        seq_pred = lstm_enoder_decoder(x_, last_x, y_)
        predicted_values.append(seq_pred)
        # Calculate loss
        loss = criterion(seq_pred, y_)
        test_loss += loss
        test_loss_list.append(loss.item())

all_test_loss = test_loss / len(test_loader)
print("Each Test Loss: {:.6f}".format(all_test_loss))
print("All Test Loss:", test_loss_list)

'''



