
#importing the modules needed
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
from google.colab import files

tf.reset_default_graph()

#dataset creation
l=[i%7 for i in range(0,500)]
k=np.random.normal(loc=25,scale=4,size=(500,1))
k=k.astype(int)

#defining the next batch
class TimeSeriesData():
  def __init__(self,xmax):
    self.resolution=1
    self.xdata=l
    self.y_true=k
    self.xmax=xmax
  def nextbatch(self,steps,ret_batch_ts=False):
    r_start=np.random.randint(low=0,high=469,dtype='int')
    t_start=r_start
    batch_ts=self.xdata[t_start:t_start+steps+1]
    y_batch=self.y_true[t_start:t_start+steps+1]
    if ret_batch_ts:
      return y_batch[:-1],y_batch[1:],batch_ts
    else:
      return y_batch[:-1].reshape(1,steps,1),y_batch[1:].reshape(1,steps,1)

#plotting the data

t1=TimeSeriesData(500)
plt.plot(t1.xdata,t1.y_true)
y1,y2,ts=t1.nextbatch(30,True)
plt.plot(ts[:-1],y1,'r')

#defining the constants
num_inputs=1
num_neurons=100
num_outputs=1
learning_rate=0.0001
num_training_iterations=2000
batch_size=1



X=tf.placeholder(tf.float32,[1,30,1])

y=tf.placeholder(tf.float32,[1,30,1])

#Creating the RECURRENT Cell
cell=tf.contrib.rnn.GRUCell(num_units=1024,activation=tf.nn.relu)
cell=tf.contrib.rnn.OutputProjectionWrapper(cell,output_size=1)

#Making the cell RECURRENT
outputs,states=tf.nn.dynamic_rnn(cell,X,dtype=tf.float32)

#backpropagation through time
loss=tf.reduce_mean(tf.square(outputs-y))
optimizer=tf.train.AdamOptimizer(learning_rate=0.0005)
train=optimizer.minimize(loss)

init=tf.global_variables_initializer()

saver=tf.train.Saver()

#traing the model and saving it
with tf.Session() as sess:
  sess.run(init)
  for i in range(5000):
    x_batch,y_batch=t1.nextbatch(30)
    sess.run(train,feed_dict={X:x_batch,y:y_batch})
    if i%100==0:
      mse=loss.eval(feed_dict={X:x_batch,y:y_batch})
      print(mse)
  saver.save(sess,'./tensor_flow')

#restoring the model and predicting 7 time steps into the future
with tf.Session() as sess:
  saver.restore(sess,'./tensor_flow')
  seq=k[0:30]
  for i in range(7):
    x_batch=np.array(seq[-30:].reshape(1,30,1))
    y_pred=sess.run(outputs,feed_dict={X:x_batch})
    print("prediction for day "+ str(i+1)+ " is "+ str(int(y_pred.flatten()[-1]))+"  ")
    seq=np.append(seq,y_pred.flatten()[-1])

#ploting the values
    
x=[i for i in range(1,41)]
plt.plot(x[15:40],t1.y_true[15:40],'b')
plt.plot(x[30:37],seq[30:37],'r')

plt.plot(x[30:37],t1.y_true[30:37],'b')
plt.plot(x[30:37],seq[30:37],'r')
