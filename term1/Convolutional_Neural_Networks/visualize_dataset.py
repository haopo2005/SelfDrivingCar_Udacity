### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
import numpy as np
# Visualizations will be shown in the notebook.
%matplotlib inline

#随机展示一张图
index = random.randint(0, len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
print(y_train[index])


################################################################################
#生成样本分布的直方图
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
%matplotlib inline

def drawLabelsDistibution(data, title):
    fig, (ax) = plt.subplots(ncols=1, figsize=(20, 5))
    labels, features_per_label = np.unique(data, return_counts=True)
    ax.bar(labels, features_per_label)
    ax.set_xticks(labels)
    ax.set_ylabel('Number of samples')
    ax.set_title('Signs distribution: %s' % title)
    fig.tight_layout()
    plt.show()


# Distribution of classes in the training set
drawLabelsDistibution(y_train, 'training')
# Distribution of classes in the validation set
drawLabelsDistibution(y_valid, 'validation')
# Distribution of classes in the test set
drawLabelsDistibution(y_test, 'test')


##################################################################################
#更好看的展示，带标签
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
%matplotlib inline

signnames = pd.read_csv('./signnames.csv')
signnames.set_index('ClassId',inplace=True)

def get_name_from_label(label):
    # Helper, transofrm a numeric label into the corresponding strring
    return signnames.loc[label].SignName

counter = Counter(y_train)
counts = pd.DataFrame(columns=['sign_label','training_samples_count'],data=[(label, count) for label, count in counter.items()])
counts['sign'] = counts.sign_label.apply(get_name_from_label)
plt.figure(figsize=(12,12))
sns.set(font_scale=1.3)
sns.barplot(x='training_samples_count',y='sign',data=counts,orient='o')
plt.xticks(rotation=90)
plt.ylabel('Sign Name')
plt.xlabel('# Training Samples');
plt.tight_layout()

#############################################################################
#更复杂的随机展示，展示所有类别，每列随机挑选5列
def get_random_samples(n_max=1):
    selected = list()
    
    for current_label in range(43):
        found=0
        while found<n_max:
            ii = np.random.randint(len(y_train))
            
            if current_label == y_train[ii]:
                selected.append(ii)
                found+=1


    assert len(selected) ==  43*n_max
    return list(selected)

plt.figure(figsize=(10,40))
count=0
cols = 5
for ii in get_random_samples(cols):
    count+=1
    plt.subplot(43,cols,count)
    plt.imshow(X_train[ii])
    plt.axis('off')
# plt.tight_layout()
plt.savefig('random_examples.png',bbox_inches='tight')
