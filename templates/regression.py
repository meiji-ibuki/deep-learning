import tensorflow as tf
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
X,y=make_regression(n_samples=1000,n_features=12,noise=10,random_state=42)
a,t,b,u=train_test_split(X,y,test_size=.2,random_state=42)
a,v,b,w=train_test_split(a,b,test_size=.2,random_state=43)
tf.keras.utils.set_random_seed(42)
norm=tf.keras.layers.Normalization(); norm.adapt(a.astype('float32'))
m=tf.keras.Sequential([tf.keras.Input((12,)),norm,tf.keras.layers.Dense(64,activation='relu'),tf.keras.layers.Dense(1)])
m.compile(optimizer='adam',loss='mae')
m.fit(a,b,validation_data=(v,w),epochs=100,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)])
print('test MAE',mean_absolute_error(u,m.predict(t,verbose=0).ravel()))
m.save('regression.keras')
