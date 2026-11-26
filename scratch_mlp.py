import numpy as np
class Dense:
    def __init__(self, n_in, n_out, rng, relu_next=False):
        scale=np.sqrt(2/n_in) if relu_next else np.sqrt(2/(n_in+n_out))
        self.W=rng.normal(0,scale,(n_in,n_out)); self.b=np.zeros(n_out)
    def forward(self,x):
        self.x=x
        return x@self.W+self.b
    def backward(self,d):
        self.dW=self.x.T@d; self.db=d.sum(axis=0)
        return d@self.W.T
    def parameters(self):
        return [(self.W,self.dW),(self.b,self.db)]

class ReLU:
    def forward(self,x):
        self.mask=x>0
        return np.maximum(x,0)
    def backward(self,d):
        return d*self.mask

class Tanh:
    def forward(self,x):
        self.a=np.tanh(x)
        return self.a
    def backward(self,d):
        return d*(1-self.a**2)

class Sequential:
    def __init__(self,layers):
        self.layers=layers
    def forward(self,x):
        for layer in self.layers: x=layer.forward(x)
        return x
    def backward(self,d):
        for layer in reversed(self.layers): d=layer.backward(d)
        return d
    def parameters(self):
        return [pair for layer in self.layers if isinstance(layer,Dense)
                for pair in layer.parameters()]

class MSE:
    def forward(self,z,y):
        assert z.shape==y.shape and z.size>0
        self.e=z-y
        return np.mean(self.e**2)
    def backward(self):
        return 2*self.e/self.e.size

class SoftmaxCE:
    def forward(self,z,y):
        assert z.ndim==2 and y.shape==(len(z),)
        assert np.issubdtype(y.dtype,np.integer) and np.all((y>=0)&(y<z.shape[1]))
        shifted=z-z.max(axis=1,keepdims=True)
        logp=shifted-np.log(np.exp(shifted).sum(axis=1,keepdims=True))
        self.p=np.exp(logp); self.y=y.copy()
        return -logp[np.arange(len(y)),y].mean()
    def backward(self):
        d=self.p.copy(); d[np.arange(len(self.y)),self.y]-=1
        return d/len(self.y)

class SGD:
    def __init__(self,lr=.05): self.lr=lr
    def step(self,pairs):
        for p,g in pairs: p-=self.lr*g

class Momentum:
    def __init__(self,lr=.03,beta=.9):
        self.lr=lr; self.beta=beta; self.v={}
    def step(self,pairs):
        for p,g in pairs:
            v=self.v.setdefault(id(p),np.zeros_like(p))
            v*=self.beta; v+=g; p-=self.lr*v

def gradient_check(model,loss_fn,x,y,h=1e-5):
    loss_fn.forward(model.forward(x),y)
    model.backward(loss_fn.backward())
    pairs=[(p,g.copy()) for p,g in model.parameters()]
    errors=[]
    for p,g in pairs:
        numeric=np.zeros_like(p)
        for idx in np.ndindex(p.shape):
            old=p[idx]
            try:
                p[idx]=old+h; plus=loss_fn.forward(model.forward(x),y)
                p[idx]=old-h; minus=loss_fn.forward(model.forward(x),y)
            finally: p[idx]=old
            numeric[idx]=(plus-minus)/(2*h)
        np.testing.assert_allclose(numeric,g,atol=1e-7,rtol=1e-5)
        errors.append(float(np.max(np.abs(numeric-g))))
    return errors

def fit(model,loss_fn,opt,x,y,xv,yv,epochs=100,batch_size=32,seed=0):
    rng=np.random.default_rng(seed); history=[]
    for epoch in range(epochs):
        order=rng.permutation(len(x))
        for start in range(0,len(x),batch_size):
            idx=order[start:start+batch_size]
            loss_fn.forward(model.forward(x[idx]),y[idx])
            model.backward(loss_fn.backward())
            pairs=model.parameters()
            if not all(np.isfinite(g).all() for _,g in pairs):
                raise FloatingPointError('non-finite gradient')
            opt.step(pairs)
        # epoch終了時の同じ重みで両方を測る
        train=float(loss_fn.forward(model.forward(x),y))
        val=float(loss_fn.forward(model.forward(xv),yv))
        history.append((train,val))
    return np.array(history)

def regression():
    rng=np.random.default_rng(42)
    x=rng.uniform(-2,2,(320,1)); y=np.sin(2*x)+rng.normal(0,.05,x.shape)
    model=Sequential([Dense(1,24,rng),Tanh(),Dense(24,1,rng)])
    history=fit(model,MSE(),Momentum(.015,.9),x[:240],y[:240],x[240:],y[240:],epochs=150)
    return x,y,model,history

def classification():
    rng=np.random.default_rng(7)
    x=rng.uniform(-1,1,(600,2)); y=(x[:,0]*x[:,1]>0).astype(int)
    model=Sequential([Dense(2,24,rng,True),ReLU(),Dense(24,16,rng,True),ReLU(),Dense(16,2,rng)])
    history=fit(model,SoftmaxCE(),Momentum(.025,.9),x[:400],y[:400],x[400:500],y[400:500],epochs=100)
    acc=np.mean(model.forward(x[500:]).argmax(axis=1)==y[500:])
    return x,y,model,history,acc
if __name__=='__main__':
    rng=np.random.default_rng(1); m=Sequential([Dense(2,3,rng),Tanh(),Dense(3,2,rng)])
    x=rng.normal(size=(4,2)); y=np.array([0,1,1,0]); print('check CE',gradient_check(m,SoftmaxCE(),x,y));print('check MSE',gradient_check(m,MSE(),x,np.eye(2)[y]))
    x,y,m,h=regression(); print('reg',h[0],h[-1]); x,y,m,h,a=classification();print('class',h[0],h[-1],a)
