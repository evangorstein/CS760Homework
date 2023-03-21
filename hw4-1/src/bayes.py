import numpy as np
import string


#Create a list of all characters 
chars = list(string.ascii_lowercase)
chars.append(" ")

#Languages
langs = ["e", "j", "s"]
lang_dict = {"e": 0, "j":1, "s":2}

#Get training data
n_train = 10 #Ten documents per language
X_train = np.empty([3*n_train, len(chars)], dtype = int)
y_train = np.empty(3*n_train, dtype=int)

for lang in langs:
    for i in range(n_train):
        

        file_name = f"languageID/{lang}{i}.txt"
        str = open(file_name, "r").read()

        lang_num = lang_dict[lang]
        ind = n_train * lang_num + i

        bow = [str.count(char) for char in chars]
        X_train[ind,:] = bow
        y_train[ind] = lang_num


#Document class priors:
def get_prior(y_train, alpha):

    counts = np.bincount(y_train)
    prior =  (counts + alpha) / (np.sum(counts) + len(langs)*alpha)
    return prior

prior = get_prior(y_train, 0.5)

#Conditional probabilities
def get_probs(X_train, y_train, alpha):

    theta = np.empty([len(chars), len(langs)])

    for (j, lang) in enumerate(langs):
        
        #Get character counts across all documents in class j
        class_mask = y_train == j
        class_counts = np.sum(X_train[class_mask,:], axis = 0)

        #Get probabilities with smoothing
        cond_probs = (class_counts + alpha) / (np.sum(class_counts) + len(chars)*alpha)
        theta[:,j] = cond_probs
    
    return theta
   
theta = get_probs(X_train, y_train, 0.5)
print(np.round(theta, 4))

#Make predictions for e10.txt
str = open("languageID/e18.txt", "r").read()
bow = np.array([str.count(char) for char in chars])
print(bow)

log_likes = np.sum(bow * np.log(theta).T, axis = 1)
print(log_likes)
log_likes_stand = log_likes - np.max(log_likes) #To prevent underflow
probs = np.exp(log_likes_stand)/np.sum(np.exp(log_likes_stand))
print(probs)


#Make predictions for files 10.txt through 19.txt in all languages
n_test = 10
pred_probs = np.empty([3*n_test, 3])


for lang in langs:
    for i in range(n_train, n_train+n_test):

        file_name = f"languageID/{lang}{i}.txt"
        str = open(file_name, "r").read()

        lang_num = lang_dict[lang]
        ind = n_test * lang_num + i - n_train

        bow = np.array([str.count(char) for char in chars])
        
        log_likes = np.sum(bow * np.log(theta).T, axis = 1)
        log_likes_normal = log_likes - np.max(log_likes) #To prevent underflow
        probs = np.exp(log_likes_normal)/np.sum(np.exp(log_likes_normal))
        print(probs)

        pred_probs[ind,:] = probs

print(np.round(pred_probs, 2))



        






        

