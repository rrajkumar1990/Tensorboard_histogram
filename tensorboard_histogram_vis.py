# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 00:42:38 2019

@author: rajkumar rajasekaran
"""

#lets take create a random linear data set
import numpy as np
import matplotlib.pyplot as plt
x_data = np.linspace(0.0,20.0,1000)

#jus forming a random equation with some noise
noise = np.random.randn(len(x_data))
y_true =  (10.05 * x_data ) + 5 +noise

#lets plot the input dataset
plt.scatter(x_data[:250],y_true[:250],label='linearexample')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()


#lets create our model function which takes batch size , input params , y and returns our weights and bias terms as output and our file folder for our scalr variable
import tensorflow as tf

def test_model(batch_size,x_data,y_true, path):
    

    #intilaize placeholders
    x = tf.placeholder(tf.float32,[batch_size])
    y = tf.placeholder(tf.float32,[batch_size])
    
    #randomly intializing the values for weights and bias terms
    m = tf.Variable(8.0)
    b = tf.Variable(1.0)
    
    #main layer 
    y_model = m*x + b
    
    #using rmse for this regression
    error = (tf.reduce_mean(tf.square(y-y_model)))
    
    #gradient descent would be good enuf for a basic linear equation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    
    train = optimizer.minimize(error)
    
    #initialize globale variables
    init = tf.global_variables_initializer()
    
    #for seeing a histogram value in our tensorboard we need provide a variable name and mention which
    #values tensor it belongs to in our case its "m" and "b" tensor so attaching that to our histogram variable
    weight_summary= tf.summary.histogram(name='weights',values=m)
    bias_summary= tf.summary.histogram(name='bias',values=b)
    
    
    
    
    log_dir=path
    
    with tf.Session() as sess:
        #writes the data into pur log file using Filewrriter
        writer= tf.summary.FileWriter(log_dir,sess.graph)
        sess.run(init)
        
        batches = 5000
        
        for i in range(batches):
            
            rand_ind = np.random.randint(len(x_data),size=batch_size)
            
            feed = {x:x_data[rand_ind],y:y_true[rand_ind]}
            
            sess.run(train,feed_dict=feed)
            
            #lets add our varables for which we need to see the histogram plot 
            wsummary,bsummary=(sess.run([weight_summary,bias_summary],feed_dict={x:x_data[rand_ind],y:y_true[rand_ind]}))
            
            #write the given histogram files
            writer.add_summary(wsummary,i)
            writer.add_summary(bsummary,i)

            
            if i %100==0:
                rmse=sess.run(error,feed_dict={x:x_data[rand_ind],y:y_true[rand_ind]})
                print('Epochs:{} -- Trainmse :{}'.format(i,rmse))
            
            
        model_m,model_b = sess.run([m,b])
        
    return model_m,model_b



batch_size=8
x,y, path = x_data,y_true,'./tensorboard_hist_exmaple'

m,b=test_model(batch_size,x,y,path)


#lets use the new m and b and try getting the new values

actual=y_true[100:200]
pred=x_data[100:200]*m+b

plt.scatter(x_data[100:200],actual,label='actual')
plt.scatter(x_data[100:200],pred,label='pred')
plt.title('tensorboardlinearexmaple')
plt.xlabel('X')
plt.xlabel('y')
plt.legend()


#now its time to invoke our tensorboard

#open command promt and navigate to the folder path where we have saved the log in our case its same path as the file under folder tensorboard_hist_exmaple 

#now pass the following commands in the command promit

#tensorboard --logdir=tensorboard_hist_exmaple --port 6006

#6006 is the default port of tensorboard

#once clicked we can see that the we will get a url 
#for eg  http://xyz:6006 
# copy the url and open in browser

#please click on the histogram tab and refer the screenshot attached by me



    