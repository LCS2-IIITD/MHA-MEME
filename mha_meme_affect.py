
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
import re
from nltk.tokenize import word_tokenize
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import time
import torch
import torchvision
import os
from torchvision import transforms,models
import torch.nn as nn
import xlrd
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.utils import shuffle
from torch.nn import functional as F
from torch.autograd import Variable
import bcolz
import pickle
from PIL import Image
import scipy
from scipy.sparse.csgraph import minimum_spanning_tree
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


transform = transforms.Compose([
transforms.Resize(256),                    #[2]
transforms.CenterCrop(224),                #[3]
transforms.ToTensor(),                     #[4]
transforms.Normalize(                      #[5]
mean=[0.485, 0.456, 0.406],                #[6]
std=[0.229, 0.224, 0.225]                  #[7]
 )])

#### Construct the Vocabulary ####


dataFrame = pd.read_excel('dataset/data_7000_actual.xlsx', header=None)
testdata = pd.read_excel('dataset/testdata_nonsplitted.xlsx', header=None)

dataFrame = dataFrame.values[:,[0,3,8]]


x=dataFrame[:,[0,1]]
xtest = testdata.values[:,[0,3]]

#convert the entire text in lowercase
x[:, 1] = [element.lower() if isinstance(element, str) else element for element in x[:, 1]]
xtest[:, 1] = [element.lower() if isinstance(element, str) else element for element in xtest[:, 1]]

def scrub_words(text):
    """Basic cleaning of texts"""
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    text=re.sub("(\\W)"," ",text)
    return text

x[:, 1] = [scrub_words(element) if isinstance(element, str) else element for element in x[:, 1]]
xtest[:, 1] = [scrub_words(element) if isinstance(element, str) else element for element in xtest[:, 1]]


class ConstructVocab():
    def __init__(self, sentences):
        self.sentences = sentences
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()
        
    def create_index(self):
        for s in self.sentences:
            # update with individual tokens
            self.vocab.update(str(s).split(' '))
            
        # sort the vocab
        self.vocab = sorted(self.vocab)

        # add a padding token with index 0
        self.word2idx['<pad>'] = 0
        
        # word to index mapping
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1 # +1 because of pad token
        
        # index to word mapping
        for word, index in self.word2idx.items():
            self.idx2word[index] = word  
            
# construct vocab and indexing
inputs = ConstructVocab(np.concatenate((x[:,1], xtest[:,1]), axis=0))

#### Initializing GLOVE weights ####

vectors = bcolz.open(f'glove/glove.wiki200d.dat')[:]
words = pickle.load(open(f'glove/glove.wiki_words_200d.pkl', 'rb'))
word2idx = pickle.load(open(f'glove/glove.wiki_idx_200d.pkl', 'rb'))
 
glove = {w: vectors[word2idx[w]] for w in words}

#### Model without the last LSTM ####

#### Define the hyper-parameters here 
batch_size = 4
####
no_of_splits = 6
max_len = 25
embedding_dim = 200
units = 256 #dimention of hidden layer
hidden_d = 512
hidden_d2 = 100
n_classes = 2
n_layers = 1
dropout = 0.2
learning_rate = 0.005
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_weight_gradient_humour = torch.tensor([2,1]).float().to(device)
class_weight_gradient_sarcasm = torch.tensor([2,1]).float().to(device)
class_weight_gradient_offensive = torch.tensor([1.25,2]).float().to(device)
class_weight_gradient_motivation = torch.tensor([1,1.25]).float().to(device)
atmf_dense_1 = 256
atmf_dense_2 = 64
atmf_dense_3 = 8
atmf_dense_4 = 1


#### Read the splits ####
traindata = pd.read_csv('dataset/train_splitted_all_tasks.csv', header=None).to_numpy() 
testdata = pd.read_csv('dataset/test_splitted_all_tasks.csv', header=None).to_numpy() 


x=traindata[:, 0: no_of_splits+1]

humour_y = traindata[:,16].tolist()
sarcasm_y = traindata[:,17].tolist()
offensive_y = traindata[:,18].tolist()
motivational_y = traindata[:,19].tolist()


for idx in range(len(humour_y)):
    if (humour_y[idx] == 'not_funny'):
        humour_y[idx] = 0
    if (humour_y[idx] == 'funny' or humour_y[idx] =='very_funny' or humour_y[idx] =='hilarious'):
        humour_y[idx] = 1

        

for idx in range(len(sarcasm_y)):
    if (sarcasm_y[idx] == 'not_sarcastic'):
        sarcasm_y[idx] = 0
    if (sarcasm_y[idx] == 'general' or sarcasm_y[idx] =='twisted_meaning' or sarcasm_y[idx] =='very_twisted'):
        sarcasm_y[idx] = 1    

   
    
for idx in range(len(offensive_y)):
    if (offensive_y[idx] == 'not_offensive'):
        offensive_y[idx] = 0
    if (offensive_y[idx] == 'slight' or offensive_y[idx] =='very_offensive' or offensive_y[idx] =='hateful_offensive'):
        offensive_y[idx] = 1    
        
        

for idx in range(len(motivational_y)):
    if (motivational_y[idx] == 'not_motivational'):
        motivational_y[idx] = 0
    if (motivational_y[idx] == 'motivational'):
        motivational_y[idx] = 1 

        
        
xtest = testdata[:, 0: no_of_splits+1]
ytest = testdata[:, 16]
ytest = [str(ytest[idx]).zfill(4) for idx in range(len(ytest))]

ytest_humour = []
ytest_sarcasm = []
ytest_offensive = []
ytest_motivational = []

for idx in range(len(ytest)):
    ytest_humour.append(int(ytest[idx][0]))
    ytest_sarcasm.append(int(ytest[idx][1]))
    ytest_offensive.append(int(ytest[idx][2]))
    ytest_motivational.append(int(ytest[idx][3]))



train_images = x[:,0]
test_images= xtest[:,0]


#convert the entire text in lowercase
for idx in range(1, 1 + no_of_splits):
    x[:, idx] = [element.lower() if isinstance(element, str) else element for element in x[:,idx]]
    xtest[:, idx] = [element.lower() if isinstance(element, str) else element for element in xtest[:,idx]]

    

def scrub_words(text):
    """Basic cleaning of texts"""
    # remove html markup
    text=re.sub("(<.*?>)"," ",text)
    text=re.sub("(\\W)"," ",text)
    return text

for idx in range(1, 1 + no_of_splits):
    x[:, idx] = [scrub_words(element) if isinstance(element, str) else element for element in x[:, idx]]
    xtest[:, idx] = [scrub_words(element) if isinstance(element, str) else element for element in xtest[:, idx]]


# vectorize to tensor
input_tensor = []
for idx in range(len(x)):
    input_tensor_sample = [[inputs.word2idx.get(s) for s in str(es).split(' ')]  for es in x[idx,1:]]
    input_tensor.append(input_tensor_sample)


input_tensor_test = []
for idx in range(len(xtest)):
    input_tensor_testsample = [[inputs.word2idx.get(s) for s in str(es).split(' ')]  for es in xtest[idx,1:]]
    input_tensor_test.append(input_tensor_testsample)


#### To represent out-of-vocabulaary words by a space ####
for idx in range(len(input_tensor)):
    for i in range(no_of_splits):
        for v in range(len(input_tensor[idx][i])):
            if (input_tensor[idx][i][v] == None):
                input_tensor[idx][i][v] = 1

for idx in range(len(input_tensor_test)):
    for i in range(no_of_splits):
        for v in range(len(input_tensor_test[idx][i])):
            if (input_tensor_test[idx][i][v] == None):
                input_tensor_test[idx][i][v] = 1
    

#### padding to max_len ####    
def pad_sequences(x, max_len):
    padded = np.zeros((max_len), dtype=np.int64)
    if len(x) > max_len: padded[:] = x[:max_len]
    else: padded[:len(x)] = x
    return padded


# inplace padding
input_tensor_pad = []
input_tensor_test_pad = []
for idx in range(len(input_tensor)):
    xsample = [pad_sequences(x, max_len) for x in input_tensor[idx]]
    input_tensor_pad.append(xsample)

for idx in range(len(input_tensor_test)):
    xsampletest = [pad_sequences(x, max_len) for x in input_tensor_test[idx]]
    input_tensor_test_pad.append(xsampletest)


print(len(input_tensor_test_pad))
print('--------------------------------------')



#### Define the custom dataset
class MemoDataset(Dataset):

    def __init__(self, imagelist, input_ids, y_humourlist, y_sarcasmlist, y_offensivelist, y_motivationlist, root_dir, transform=None):
        
        self.imagelist = imagelist
        self.input_ids = input_ids
        self.humourlist = y_humourlist
        self.sarcasmlist = y_sarcasmlist
        self.offensivelist = y_offensivelist
        self.motivationlist = y_motivationlist
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imagelist)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.imagelist[idx])
        image = Image.open(img_name).convert('RGB')
        input_id = self.input_ids[idx]
        
        target_humour = self.humourlist[idx]
        target_sarcasm = self.sarcasmlist[idx]
        target_offensive = self.offensivelist[idx]
        target_motivation = self.motivationlist[idx]
        

        if self.transform:
            sample = self.transform(image)
          
        return (sample, input_id, target_humour, target_sarcasm, target_offensive, target_motivation)
    
    
train_data = MemoDataset(train_images, input_tensor_pad, humour_y, sarcasm_y, offensive_y, motivational_y, root_dir = 'dataset/train_meme_images', transform=transform)
test_data = MemoDataset(test_images, input_tensor_test_pad, ytest_humour, ytest_sarcasm, ytest_offensive, ytest_motivational, root_dir = 'dataset/test_meme_images', transform=transform) 
    
trainloader = DataLoader(train_data, batch_size=batch_size, drop_last = True, shuffle=True, num_workers=4)    
testloader = DataLoader(test_data, batch_size=batch_size, drop_last=True, shuffle=True, num_workers=4)

vocab_inp_size = len(inputs.word2idx)

#####
# Here I use the Glove Embedding for the vocabulary of this training and testing set
##### 

matrix_len = len(inputs.word2idx)
weights_matrix = np.zeros((matrix_len, 200))
words_found = 0

for i, word in enumerate(inputs.vocab):
    try: 
        weights_matrix[i] = glove[word]
        words_found += 1
    except KeyError:
        weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

print('---------------------------')
print(f'Number of Word Embeddings replaced by Pre-Trained Glove: {words_found}')
print('---------------------------')    



class VGG19Bottom(nn.Module):
    def __init__(self, original_model):
        super(VGG19Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-2])
        
    def forward(self, x):
        x = self.features(x)
        return x


class MemoLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, vocab_size, embedding_length, hidden_d, dropout, n_layers):
        super(MemoLSTM, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length
        self.hidden_d = hidden_d


        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
        self.word_embeddings.weight.data.copy_(torch.from_numpy(weights_matrix))
        
        self.bilstm = nn.LSTM(embedding_length, hidden_size, dropout=dropout, num_layers = n_layers, bidirectional=True)
        # We will use da = 350, r = 30 & penalization_coeff = 1 as per given in the self-attention original ICLR paper
        
        
        self.vgg19 = models.vgg19(pretrained=True)
        self.vgg19bottom = VGG19Bottom(self.vgg19).to(device)
        self.drop = nn.Dropout(p=0.2)
        
        self.l_1 = nn.Linear(2*units*30, hidden_d)
        
        self.W_b = nn.Parameter(torch.rand(batch_size, 512, 512))
        
        self.W_s1 = nn.Linear(512, 350) #### 2*units = 512
        self.W_s2 = nn.Linear(350, 30)
        
        self.atmf_dense_1 = nn.Linear(hidden_d, atmf_dense_1)
        self.atmf_dense_2 = nn.Linear(atmf_dense_1, atmf_dense_2)
        self.atmf_dense_3 = nn.Linear(atmf_dense_2, atmf_dense_3)
        self.atmf_dense_4 = nn.Linear(atmf_dense_3, atmf_dense_4)
        self.W_F = nn.Parameter(torch.rand(batch_size, 512, 512))
        self.W_f = nn.Parameter(torch.rand(batch_size, 512, 1))
        
      
    
    def image_encoding_filter(self, text_features, img_features):
        
        #### text_features - (batch_size, num_seq_n, 512)
        #### img_features - (batch_size, num_seq_m, 512)
        img_features_tran = img_features.permute(0, 2, 1)
        affinity_matrix_int = torch.bmm(text_features, self.W_b)
        affinity_matrix = torch.bmm(affinity_matrix_int, img_features_tran)
        
        affinity_matrix_sum = torch.sum(affinity_matrix, dim=1)
        affinity_matrix_sum = torch.unsqueeze(affinity_matrix_sum, dim=1)
        alpha_h = affinity_matrix/affinity_matrix_sum

        alpha_h_tran = alpha_h.permute(0,2,1)
        a_h = torch.bmm(alpha_h_tran, text_features)

        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        gates = (1 - cos(img_features.cpu(), a_h.cpu())).to(device)

        gated_image_features = a_h * gates[:, :, None]     
        
        return gated_image_features
        

    def atmf(self, mm_feature):

        mm_feature_tran = mm_feature.permute(0,2,1)
        s = self.atmf_dense_4(self.atmf_dense_3(self.atmf_dense_2(self.atmf_dense_1(mm_feature_tran))))
        s = s.permute(0,2,1)
        
        s = F.softmax(s, dim=2) + 1
    
        wei_fea  = mm_feature * s
        P_F = torch.tanh(torch.bmm(self.W_F, wei_fea))
        P_F = F.softmax(P_F, dim=2)
        
        
        gamma_f = torch.bmm(self.W_f.permute(0,2,1),P_F)
        gamma_f = gamma_f.permute(0,2,1)
        atmf_output = torch.bmm(wei_fea,gamma_f)
        

        return atmf_output
    
     
    
    
    def attention_net(self, lstm_output):

        attn_weight_matrix = self.W_s2(torch.tanh(self.W_s1(lstm_output)))
        attn_weight_matrix = attn_weight_matrix.permute(0, 2, 1)
        attn_weight_matrix = F.softmax(attn_weight_matrix, dim=2)

        return attn_weight_matrix
    

    
    
    def forward(self, img, input_sentences):
 
        input_split = self.word_embeddings(input_sentences)
        input_split = input_split.permute(1,0,2)

        output, (h_n, c_n) = self.bilstm(input_split)
        output = output.permute(1, 0, 2)
        #print(output.shape)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        # h_n.size() = (1, batch_size, hidden_size)
        # c_n.size() = (1, batch_size, hidden_size)

        attn_weight_matrix = self.attention_net(output)
        # attn_weight_matrix.size() = (batch_size, r, num_seq)
        # output.size() = (batch_size, num_seq, 2*hidden_size)
        hidden_matrix = torch.bmm(attn_weight_matrix, output)
        # hidden_matrix.size() = (batch_size, r, 2*hidden_size)
        hidden_matrix = hidden_matrix.view(-1, hidden_matrix.size()[1] * hidden_matrix.size()[2])


        img_features = self.vgg19bottom(img)
        img_features = img_features.reshape(batch_size,512,49)
        img_features = img_features.permute(0, 2, 1)
        
        img_features = self.image_encoding_filter(output, img_features)
        
        attn_weight_matrix = self.attention_net(img_features)
        atten_img_features = torch.bmm(attn_weight_matrix, img_features)
        atten_img_features = atten_img_features.view(-1, atten_img_features.size()[1] * atten_img_features.size()[2])

        # Let's now concatenate the hidden_matrix, apply atmf and connect it to the fully connected layer.
        s_text = self.l_1(hidden_matrix)
        s_image = self.l_1(atten_img_features)

        mm_features = torch.stack((s_text,s_image)).permute(1,2,0)
        

        atmf_output = self.atmf(mm_features)
        atmf_output = torch.squeeze(atmf_output,2)
        
        return atmf_output

        
        
class ClassifierLSTM(nn.Module):
    def __init__(self, hidden_d, dropout, n_layers):
        super(ClassifierLSTM, self).__init__()
        
        
        self.hidden_d = hidden_d
        
        self.prev_out_humour = nn.Linear(no_of_splits*hidden_d, hidden_d2)
        self.out_humour = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_sarcasm = nn.Linear(no_of_splits*hidden_d, hidden_d2)
        self.out_sarcasm = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_offensive = nn.Linear(no_of_splits*hidden_d, hidden_d2)
        self.out_offensive = nn.Linear(hidden_d2, n_classes)
        
        self.prev_out_motivation = nn.Linear(no_of_splits*hidden_d, hidden_d2)
        self.out_motivation = nn.Linear(hidden_d2, n_classes)
        
        self.drop = nn.Dropout(p=0.2)
        
    
    def forward(self, fc_input):

        
        fc_input = fc_input.permute(1, 0, 2)
        mixed_features = fc_input.contiguous().view(-1, fc_input.size()[1] * fc_input.size()[2])
        
        logits_humour = self.prev_out_humour(mixed_features)
        logits_humour = self.drop(logits_humour)
        logits_humour = self.out_humour(logits_humour)
        logits_humour = self.drop(logits_humour)
        
        logits_sarcasm = self.prev_out_sarcasm(mixed_features)
        logits_sarcasm = self.drop(logits_sarcasm)
        logits_sarcasm = self.out_sarcasm(logits_sarcasm)
        logits_sarcasm = self.drop(logits_sarcasm)
        
        logits_offensive = self.prev_out_offensive(mixed_features)
        logits_offensive = self.drop(logits_offensive)
        logits_offensive = self.out_offensive(logits_offensive)
        logits_offensive = self.drop(logits_offensive)
        
        logits_motivation = self.prev_out_motivation(mixed_features)
        logits_motivation = self.drop(logits_motivation)
        logits_motivation = self.out_motivation(logits_motivation)
        logits_motivation = self.drop(logits_motivation)

        return (logits_humour, logits_sarcasm, logits_offensive, logits_motivation)

    
### Enabling CUDA
model = MemoLSTM(n_classes, units, vocab_inp_size, embedding_dim, hidden_d, dropout, n_layers)
classifier = ClassifierLSTM(hidden_d, dropout, n_layers)

model.to(device)
classifier.to(device)
params = list(model.parameters()) + list(classifier.parameters())

# Loss and optimizer
criterion_humour = nn.CrossEntropyLoss(weight = class_weight_gradient_humour)
criterion_sarcasm = nn.CrossEntropyLoss(weight = class_weight_gradient_sarcasm)
criterion_offensive = nn.CrossEntropyLoss(weight = class_weight_gradient_offensive)
criterion_motivation = nn.CrossEntropyLoss(weight = class_weight_gradient_motivation)

optimizer = torch.optim.Adam(params, lr = learning_rate, weight_decay=1e-4)


batch_wise_loss = []
batch_wise_micro_f1_humour = []
batch_wise_macro_f1_humour = []
epoch_wise_macro_f1_humour = []
epoch_wise_micro_f1_humour = []

batch_wise_micro_f1_sarcasm = []
batch_wise_macro_f1_sarcasm = []
epoch_wise_macro_f1_sarcasm = []
epoch_wise_micro_f1_sarcasm = []

batch_wise_micro_f1_offensive = []
batch_wise_macro_f1_offensive = []
epoch_wise_macro_f1_offensive = []
epoch_wise_micro_f1_offensive = []

batch_wise_micro_f1_motivation = []
batch_wise_macro_f1_motivation = []
epoch_wise_macro_f1_motivation = []
epoch_wise_micro_f1_motivation = []

epoch_wise_average_macro_f1 = []

# Train the model
n_total_steps = len(trainloader)
for epoch in range(epochs):
    
    target_total_humour = []
    predicted_total_humour = []
    
    target_total_sarcasm = []
    predicted_total_sarcasm = []
    
    target_total_offensive = []
    predicted_total_offensive = []
    
    target_total_motivation = []
    predicted_total_motivation = []
    
    for i, (images, input_ids, labels_humour, labels_sarcasm, labels_offensive, labels_motivation) in enumerate(trainloader):  
        
        model.train()
        classifier.train()
        # Forward pass
        images = images.to(device) 
        labels_humour = labels_humour.to(device)
        labels_sarcasm = labels_sarcasm.to(device)
        labels_offensive = labels_offensive.to(device)
        labels_motivation = labels_motivation.to(device)
        
        
        mm_atmf_outputs = []
        
        for idx in range(no_of_splits):
            input_utt = torch.tensor(input_ids[idx]).to(device)
            atmf_output = model(images, input_utt)
            mm_atmf_outputs.append(atmf_output)
        
        lstm_input = torch.stack(mm_atmf_outputs, dim=0)
        outputs_humour, outputs_sarcasm, outputs_offensive, outputs_motivation = classifier(lstm_input)
        

        loss_humour = criterion_humour(outputs_humour, labels_humour)
        loss_sarcasm = criterion_sarcasm(outputs_sarcasm, labels_sarcasm)
        loss_offensive = criterion_offensive(outputs_offensive, labels_offensive)
        loss_motivation = criterion_motivation(outputs_motivation, labels_motivation)
        loss_average = (loss_humour + loss_sarcasm + loss_offensive + loss_motivation)/4
        
        # Backward and optimize
        optimizer.zero_grad()
        loss_average.backward()
        optimizer.step()
        
        # max returns (value ,index)
        _, predicted_humour = torch.max(outputs_humour.data, 1)
        _, predicted_sarcasm = torch.max(outputs_sarcasm.data, 1)
        _, predicted_offensive = torch.max(outputs_offensive.data, 1)
        _, predicted_motivation = torch.max(outputs_motivation.data, 1)
        
        target_total_humour.append(labels_humour)
        predicted_total_humour.append(predicted_humour)
        
        target_total_sarcasm.append(labels_sarcasm)
        predicted_total_sarcasm.append(predicted_sarcasm)
        
        target_total_offensive.append(labels_offensive)
        predicted_total_offensive.append(predicted_offensive)
        
        target_total_motivation.append(labels_motivation)
        predicted_total_motivation.append(predicted_motivation)
        
        
        
        target_inter_humour = [t.cpu().numpy() for t in target_total_humour]
        predicted_inter_humour = [t.cpu().numpy() for t in predicted_total_humour]
        target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
        predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
        
        target_inter_sarcasm = [t.cpu().numpy() for t in target_total_sarcasm]
        predicted_inter_sarcasm = [t.cpu().numpy() for t in predicted_total_sarcasm]
        target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
        predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
        
        target_inter_offensive = [t.cpu().numpy() for t in target_total_offensive]
        predicted_inter_offensive = [t.cpu().numpy() for t in predicted_total_offensive]
        target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
        predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
        
        target_inter_motivation = [t.cpu().numpy() for t in target_total_motivation]
        predicted_inter_motivation = [t.cpu().numpy() for t in predicted_total_motivation]
        target_inter_motivation =  np.stack(target_inter_motivation, axis=0).ravel()
        predicted_inter_motivation =  np.stack(predicted_inter_motivation, axis=0).ravel()
        
        
        
        batch_wise_loss.append(loss_average.item())
        
        batch_wise_micro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="micro"))
        batch_wise_macro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="macro"))
        
        batch_wise_micro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro"))
        batch_wise_macro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro"))
        
        batch_wise_micro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="micro"))
        batch_wise_macro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="macro"))
        
        batch_wise_micro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="micro"))
        batch_wise_macro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="macro"))
        
        
        if (i+1) % 200 == 0:
            print (f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss_average.item():.4f}')
            print(f' Humour Micro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_humour, predicted_inter_humour, average="micro")}')
            print(f' Humour Macro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_humour, predicted_inter_humour, average="macro")}')
            
            print(f' Sarcasm Micro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro")}')
            print(f' Sarcasm Macro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro")}')
            
            print(f' Offensive Micro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_offensive, predicted_inter_offensive, average="micro")}')
            print(f' Offensive Macro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_offensive, predicted_inter_offensive, average="macro")}')
        
            print(f' Motivation Micro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_motivation, predicted_inter_motivation, average="micro")}')
            print(f' Motivation Macro F1 on the training set after batch no {i+1}, Epoch [{epoch+1}/{epochs}]: {f1_score(target_inter_motivation, predicted_inter_motivation, average="macro")}')
            
            
    target_total_test_humour = []
    predicted_total_test_humour = []
    
    target_total_test_sarcasm = []
    predicted_total_test_sarcasm = []
    
    target_total_test_offensive = []
    predicted_total_test_offensive = []
    
    target_total_test_motivation = []
    predicted_total_test_motivation = [] 

    # Test the model
    # In test phase, we don't need to compute gradients (for memory efficiency)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for i_batch, (images, input_ids, labels_humour, labels_sarcasm, labels_offensive, labels_motivation) in enumerate(testloader):
            model.eval()
            classifier.eval()
            images = images.to(device)
            labels_humour = labels_humour.to(device)
            labels_sarcasm = labels_sarcasm.to(device)
            labels_offensive = labels_offensive.to(device)
            labels_motivation = labels_motivation.to(device)

            mm_atmf_outputs = []

            for idx in range(no_of_splits):
                input_utt = torch.tensor(input_ids[idx]).to(device)
                atmf_output = model(images, input_utt)
                mm_atmf_outputs.append(atmf_output)

            lstm_input = torch.stack(mm_atmf_outputs, dim=0)
            outputs_humour, outputs_sarcasm, outputs_offensive, outputs_motivation = classifier(lstm_input)
                
            # max returns (value ,index)
            
            _, predicted_humour = torch.max(outputs_humour.data, 1)
            _, predicted_sarcasm = torch.max(outputs_sarcasm.data, 1)
            _, predicted_offensive = torch.max(outputs_offensive.data, 1)
            _, predicted_motivation = torch.max(outputs_motivation.data, 1)
            
            
            
            target_total_test_humour.append(labels_humour)
            predicted_total_test_humour.append(predicted_humour)
        
            target_total_test_sarcasm.append(labels_sarcasm)
            predicted_total_test_sarcasm.append(predicted_sarcasm)
        
            target_total_test_offensive.append(labels_offensive)
            predicted_total_test_offensive.append(predicted_offensive)
        
            target_total_test_motivation.append(labels_motivation)
            predicted_total_test_motivation.append(predicted_motivation)
            
            

            target_inter_humour = [t.cpu().numpy() for t in target_total_test_humour]
            predicted_inter_humour = [t.cpu().numpy() for t in predicted_total_test_humour]
            target_inter_humour =  np.stack(target_inter_humour, axis=0).ravel()
            predicted_inter_humour =  np.stack(predicted_inter_humour, axis=0).ravel()
        
            target_inter_sarcasm = [t.cpu().numpy() for t in target_total_test_sarcasm]
            predicted_inter_sarcasm = [t.cpu().numpy() for t in predicted_total_test_sarcasm]
            target_inter_sarcasm =  np.stack(target_inter_sarcasm, axis=0).ravel()
            predicted_inter_sarcasm =  np.stack(predicted_inter_sarcasm, axis=0).ravel()
        
            target_inter_offensive = [t.cpu().numpy() for t in target_total_test_offensive]
            predicted_inter_offensive = [t.cpu().numpy() for t in predicted_total_test_offensive]
            target_inter_offensive =  np.stack(target_inter_offensive, axis=0).ravel()
            predicted_inter_offensive =  np.stack(predicted_inter_offensive, axis=0).ravel()
        
            target_inter_motivation = [t.cpu().numpy() for t in target_total_test_motivation]
            predicted_inter_motivation = [t.cpu().numpy() for t in predicted_total_test_motivation]
            target_inter_motivation =  np.stack(target_inter_motivation, axis=0).ravel()
            predicted_inter_motivation =  np.stack(predicted_inter_motivation, axis=0).ravel()

            
        current_macro_humour = f1_score(target_inter_humour, predicted_inter_humour, average="macro")
        current_macro_sarcasm = f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro")
        current_macro_offensive = f1_score(target_inter_offensive, predicted_inter_offensive, average="macro")
        current_macro_motivation = f1_score(target_inter_motivation, predicted_inter_motivation, average="macro")
        
        current_micro_humour = f1_score(target_inter_humour, predicted_inter_humour, average="micro")
        current_micro_sarcasm = f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro")
        current_micro_offensive = f1_score(target_inter_offensive, predicted_inter_offensive, average="micro")
        current_micro_motivation = f1_score(target_inter_motivation, predicted_inter_motivation, average="micro")
        
        current_macro_average = (current_macro_humour + current_macro_sarcasm + current_macro_offensive + current_macro_motivation)/4 
        epoch_wise_average_macro_f1.append(current_macro_average)
        
        epoch_wise_micro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="micro"))
        epoch_wise_macro_f1_humour.append(f1_score(target_inter_humour, predicted_inter_humour, average="macro"))
        
        epoch_wise_micro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="micro"))
        epoch_wise_macro_f1_sarcasm.append(f1_score(target_inter_sarcasm, predicted_inter_sarcasm, average="macro"))
        
        epoch_wise_micro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="micro"))
        epoch_wise_macro_f1_offensive.append(f1_score(target_inter_offensive, predicted_inter_offensive, average="macro"))
        
        epoch_wise_micro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="micro"))
        epoch_wise_macro_f1_motivation.append(f1_score(target_inter_motivation, predicted_inter_motivation, average="macro"))

        
        print(f' Humour Micro F1 on test set: {current_micro_humour}')
        print(f' Humour Macro F1 on test set: {current_macro_humour}')
        print(f' Sarcasm Micro F1 on test set: {current_micro_sarcasm}')
        print(f' Sarcasm Macro F1 on test set: {current_macro_sarcasm}')
        print(f' Offensive Micro F1 on test set: {current_micro_offensive}')
        print(f' Offensive Macro F1 on test set: {current_macro_offensive}')
        print(f' Motivation Micro F1 on test set: {current_micro_motivation}')
        print(f' Motivation Macro F1 on test set: {current_macro_motivation}')
        print(f' Current Average Macro F1 on test set: {current_macro_average}')
        
        
        print(confusion_matrix(target_inter_humour, predicted_inter_humour))
        print(confusion_matrix(target_inter_sarcasm, predicted_inter_sarcasm))
        print(confusion_matrix(target_inter_offensive, predicted_inter_offensive))
        print(confusion_matrix(target_inter_motivation, predicted_inter_motivation))
        
        print(f'Best Macro F1 on test set till this epoch: {max(epoch_wise_average_macro_f1)} Found in Epoch No: {epoch_wise_average_macro_f1.index(max(epoch_wise_average_macro_f1))+1}')
    
           

