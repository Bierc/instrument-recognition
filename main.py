import sound_in
import feat
import pickle

import pandas as pd

with open('random_search_RF.pickle', 'rb') as file:
    seu_modelo = pickle.load(file)

sound_in.sound_in()
features = feat.feature_extract('teste.wav')
dict_test = {}
dict_test[file] = features

#convert dict to dataframe
features_test = pd.DataFrame.from_dict(dict_test, orient='index',
                                       columns=['harmonic', 'mfcc', 'spectro', 'chroma', 'contrast'])


#extract mfccs
mfcc_test = pd.DataFrame(features_test.mfcc.values.tolist(),index=features_test.index)
mfcc_test = mfcc_test.add_prefix('mfcc_')

#extract spectro
spectro_test = pd.DataFrame(features_test.spectro.values.tolist(),index=features_test.index)
spectro_test = spectro_test.add_prefix('spectro_')


#extract chroma
chroma_test = pd.DataFrame(features_test.chroma.values.tolist(),index=features_test.index)
chroma_test = chroma_test.add_prefix('chroma_')


#extract contrast
contrast_test = pd.DataFrame(features_test.contrast.values.tolist(),index=features_test.index)
contrast_test = chroma_test.add_prefix('contrast_')

#drop the old columns
features_test = features_test.drop(labels=['mfcc', 'spectro', 'chroma', 'contrast'], axis=1)

#concatenate
df_features_test=pd.concat([features_test, mfcc_test, spectro_test, chroma_test, contrast_test],
                           axis=1, join='inner')

targets_test = []

targets_test.append('test')
df_features_test['targets'] = targets_test
X_test = df_features_test.drop(labels=['targets'], axis=1)

result = seu_modelo.predict(X_test)

if(result == 0):
    print("Bass")
elif(result == 1):
    print("Brass")
elif(result == 2):
    print("Flute")
elif(result == 3):
    print("Guitar")
elif(result == 4):
    print("Keyboard")
elif(result == 5):
    print("Mallet")
elif(result == 6):
    print("Organ")
elif(result == 7):
    print("Reed")
elif(result == 8):
    print("String")
elif(result == 9):
    print("Synth Lead")
elif(result == 10):
    print("Vocal")
else:
    print("Erro")


print(result)


