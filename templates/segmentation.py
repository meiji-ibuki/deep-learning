import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as L
tf.keras.utils.set_random_seed(42)
rng=np.random.default_rng(42); yy,xx=np.mgrid[:32,:32]
mask=np.stack([(xx-a)**2+(yy-b)**2<r*r for a,b,r in zip(rng.integers(8,24,600),rng.integers(8,24,600),rng.integers(3,8,600))])[...,None].astype('float32')
x=np.clip(mask+rng.normal(0,.3,mask.shape),0,1).astype('float32')
i=L.Input((32,32,1)); s=L.Conv2D(16,3,padding='same',activation='relu')(i)
z=L.MaxPooling2D()(s); z=L.Conv2D(32,3,padding='same',activation='relu')(z)
z=L.UpSampling2D()(z); z=L.Concatenate()([z,s]); z=L.Conv2D(16,3,padding='same',activation='relu')(z)
m=tf.keras.Model(i,L.Conv2D(1,1,activation='sigmoid')(z))
m.compile(optimizer='adam',loss='binary_crossentropy',metrics=[tf.keras.metrics.BinaryIoU(target_class_ids=[1],threshold=.5)])
m.fit(x[:400],mask[:400],validation_data=(x[400:500],mask[400:500]),epochs=12,verbose=0,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
print(m.evaluate(x[500:],mask[500:])); m.save('segment.keras')
