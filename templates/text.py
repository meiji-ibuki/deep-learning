import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers as L
tf.keras.utils.set_random_seed(42)
(x,y),(xt,yt)=tf.keras.datasets.imdb.load_data(num_words=10000)
x=tf.keras.utils.pad_sequences(x,maxlen=200,padding='post',truncating='post')
xt=tf.keras.utils.pad_sequences(xt,maxlen=200,padding='post',truncating='post')
a,v,b,w=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
m=tf.keras.Sequential([L.Input((200,),dtype='int32'),L.Embedding(10000,64,mask_zero=True),L.Bidirectional(L.GRU(32)),L.Dense(1,activation='sigmoid')])
m.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
m.fit(a,b,validation_data=(v,w),epochs=8,batch_size=128,callbacks=[tf.keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True)])
print(m.evaluate(xt,yt)); m.save('text.keras')
