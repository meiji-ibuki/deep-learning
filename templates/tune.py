import tensorflow as tf
import keras_tuner as kt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
x,y=load_digits(return_X_y=True); x=x.astype('float32')/16
x,t,y,u=train_test_split(x,y,test_size=.2,stratify=y,random_state=42)
a,v,b,w=train_test_split(x,y,test_size=.2,stratify=y,random_state=43)
tf.keras.utils.set_random_seed(42)
def build(hp):
    m=tf.keras.Sequential([tf.keras.Input((64,)),tf.keras.layers.Dense(hp.Choice('width',[32,64,128]),activation='relu'),tf.keras.layers.Dropout(hp.Float('dropout',0,.4,step=.2)),tf.keras.layers.Dense(10,activation='softmax')])
    m.compile(optimizer=tf.keras.optimizers.Adam(hp.Float('lr',1e-4,1e-2,sampling='log')),loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return m
search=kt.RandomSearch(build,objective='val_loss',max_trials=8,seed=42,directory='tuning',project_name='digits',overwrite=True)
search.search(a,b,validation_data=(v,w),epochs=20,callbacks=[tf.keras.callbacks.EarlyStopping(patience=3)])
print(search.get_best_hyperparameters(1)[0].values)
model=search.get_best_models(1)[0]
print(model.evaluate(t,u)); model.save('tuned.keras')
