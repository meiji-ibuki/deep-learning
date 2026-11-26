import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
tf.keras.utils.set_random_seed(42)
rng=np.random.default_rng(42); t=np.arange(2000)
y=(10+np.sin(t*2*np.pi/24)+.002*t+rng.normal(0,.2,len(t))).astype('float32')
mu,sigma=y[:1400].mean(),y[:1400].std()
z=(y-mu)/sigma
lookback,h=48,12
starts=np.arange(lookback,len(y)-h+1)
X=np.array([z[i-lookback:i] for i in starts])[...,None]
Y=np.array([z[i:i+h] for i in starts])
a=starts+h<=1400; b=(starts>=1400)&(starts+h<=1700); c=starts>=1700
m=tf.keras.Sequential([L.Input((lookback,1)),L.GRU(32),L.Dense(h)])
m.compile(optimizer='adam',loss='mae')
m.fit(X[a],Y[a],validation_data=(X[b],Y[b]),epochs=30,shuffle=False,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=4,restore_best_weights=True)])
p=m.predict(X[c],verbose=0)
print('MAE',np.mean(abs(p-Y[c]))*sigma,'seasonal naive',np.mean(abs(X[c,-24:-12,0]-Y[c]))*sigma)
