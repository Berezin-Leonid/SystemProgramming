import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = 'ptbxl/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))


'''
y_test = Y[Y.strat_fold == 10]
print(y_test.scp_codes[9])
x = y_test.scp_codes[9]

for x in y_test.scp_codes:
    if ('AFIB' in x):
        print(x)
print(Y.scp_codes)
'''

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
test_fold = 10
# Train
X_train = X[np.where(Y.strat_fold != test_fold)]
y = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
y_train = []
for x in y:
    if ('AFIB' in x):
        y_train.append(1)
    else:
        y_train.append(0)
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y = Y[Y.strat_fold == test_fold].diagnostic_superclass
y_test = []
for x in y:
    if ('AFIB' in x):
        y_test.append(1)
    else:
        y_test.append(0)

'''
print(y_test[1])
exit()
'''

# Construction of CNN
class ResidialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(ResidialBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNormalization(out_channel),
            nn.ReLu()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNormalization(out_channel),
        )
        self.relu = nn.ReLu()
    def forward(self, x):
        out = conv1(x)
        out = conv2(x)
        out += x
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=7, stride = 2, padding = 3),
            BatchNormalization(64),
            nn.ReLu()
        )
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avgpool = nn.AvgPool1d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim = 1)


    def _make_layer(self, block, planes, blocks, stride = 1):
        layers = []

        for i in range(blocks):
            layers.append(block(self.inplanes, planes))
            self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        #x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def predict(self,x):
        x = self.forward(x)
        x = self.softmax(x) 
        return x
        

# Learning
resnet = ResNet1d()

num_classes = 1
num_epochs = 20
batch_size = 100
model = ResNet1d(ResidualBlock, [3,4,6,3])
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index+batch_size]

        X_batch = X_train[batch_indexes] #.to(device)
        y_batch = y_train[batch_indexes] #.to(device)
 
        preds = resnet.forward(X_batch) 

        loss_value = criterion(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(X_test)
#     test_loss_history.append(loss(test_preds, y_test))
    
    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
#     test_accuracy_history.append(accuracy)
    print(accuracy)

# Download net
