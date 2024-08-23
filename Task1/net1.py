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
#path = '/SystemProgramming/Task1/ptbxl/'
sampling_rate=100

# load and convert annotation data
print(path+'ptbxl_database.csv')
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))



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
X_train = torch.tensor(X_train).float().permute(0,2,1)
y = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
y_train = []
for x in y:
    if ('AFIB' in x):
        y_train.append(1)
    else:
        y_train.append(0)
y_train = torch.tensor(y_train)


# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
X_test = torch.tensor(X_test).float().permute(0,2,1)
y = Y[Y.strat_fold == test_fold].diagnostic_superclass
y_test = []
for x in y:
    if ('AFIB' in x):
        y_test.append(1)
    else:
        y_test.append(0)
y_test = torch.tensor(y_test)



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channel),
        )
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channel = out_channel
        
    def forward(self, input):
        residual = input
        output = self.conv1(input)
        output = self.conv2(output)
        #pdb.set_trace()
        if (self.downsample):
            residual = self.downsample(input)
            
        output = output + residual
        output = self.relu(output)
        return output

class ResNet1d(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet1d, self).__init__()

        self.inplanes = 64
        self.conv1 = nn.Sequential(
            nn.Conv1d(12, 64, kernel_size=7, stride = 2, padding = 3),
            nn.BatchNorm1d(64),
            nn.ReLU()
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
        downsample = None
        if stride != 1 or self.inplanes != planes:
            #pdb.set_trace()
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
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
import pdb

resnet = ResNet1d(ResidualBlock, [3, 4, 6, 3])

num_classes = 1
num_epochs = 20
batch_size = 100
model = ResNet1d(ResidualBlock, [3,4,6,3])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(num_epochs):
    order = np.random.permutation(len(X_train))

    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index+batch_size]

        X_batch = X_train[batch_indexes] #.to(device)
        y_batch = y_train[batch_indexes] #.to(device)
        #pdb.set_trace()
        preds = resnet.forward(X_batch)

        loss_value = criterion(preds, y_batch)
        loss_value.backward()

        optimizer.step()

    test_preds = mnist_net.forward(X_test)
#     test_loss_history.append(loss(test_preds, y_test))

    accuracy = (test_preds.argmax(dim=1) == y_test).float().mean()
#     test_accuracy_history.append(accuracy)
    print(accuracy)

#accuracy in metrics

with torch.no_grad():
    predictions = model(X_test).squeeze()
    predicted_labels = (predictions > 0.5).float()

roc_auc = roc_auc_score(y_test, predicted_labels)
f1 = f1_score(target, predicted_labels)
sensitivity = recall_score(target, predicted_labels)

print(f'ROC-AUC: {roc_auc:.4f}')
print(f'F1-score: {f1:.4f}')
print(f'Sensitivity (Recall): {sensitivity:.4f}')


# Download net
torch.save(model.state_dict(), 'model.pth')
