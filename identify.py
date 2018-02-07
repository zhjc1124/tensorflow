from captcha.image import ImageCaptcha
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import color

alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
def random_text(text,long):
    text_return=[]
    for i in range(long):
        c=random.choice(text)
        text_return.append(c)
    return text_return
def gen_captcha_text_img(text_source,text_num):
    img=ImageCaptcha()
    captcha_text1=random_text(text_source,text_num)
    captcha_text1="".join(captcha_text1)
    captcha=img.generate(captcha_text1)
    captcha_img1=Image.open(captcha)
    captcha_img1=np.array(captcha_img1)
    return captcha_img1,captcha_text1
def get_next_batch(batch_size=64):
    def get_true_pic():
        while True:
            img, text = gen_captcha_text_img(alphabet, 4)
            if img.shape==(60,160,3):
                return img,text
    for i in range(batch_size):
        img,text=get_true_pic()#有的图片格式不对
        img=color.rgb2gray(img)
        x_batch[i,:]= img.flatten()/255
        y_batch[i,:] = text2batch(text)
    return x_batch,y_batch
def create_convolution_layer(w_alpha=0.1,b_alpha=0.1):
    pic=tf.reshape(x,[-1,img_x,img_y,1])

    w_conv1=tf.Variable(w_alpha*tf.random_normal([3,3,1,32]))
    b_conv1=tf.Variable(b_alpha*tf.random_normal([32]))
    conv1=tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pic,w_conv1,[1,1,1,1],'SAME'),b_conv1))
    conv1=tf.nn.max_pool(conv1,[1,2,2,1],[1,2,2,1],'SAME')
    conv1=tf.nn.dropout(conv1,keep_prob)

    w_conv2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_conv2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_conv2, [1, 1, 1, 1], 'SAME'), b_conv2))
    conv2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_conv3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_conv3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_conv3, [1, 1, 1, 1], 'SAME'), b_conv3))
    conv3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    w_d=tf.Variable(w_alpha*tf.random_normal([8*20*64,1024]))
    b_d=tf.Variable(b_alpha*tf.random_normal([1024]))
    dense=tf.reshape(conv3,[-1,w_d.get_shape().as_list()[0]])#这里看看是什么样子的
    dense=tf.nn.relu(tf.add(tf.matmul(dense,w_d),b_d))
    dense=tf.nn.dropout(dense,keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, text_num*source_long]))
    b_out = tf.Variable(b_alpha * tf.random_normal([text_num*source_long]))
    out=tf.add(tf.matmul(dense,w_out),b_out)

    return out
def train_model():
    output=create_convolution_layer()
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=y))
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-6).minimize(loss)
    y_pred=tf.reshape(output,[-1,text_num,source_long])
    y_pred1 = tf.reshape(output, [-1, text_num, source_long])
    y_pred=tf.argmax(y_pred,2)
    y_true=tf.argmax(tf.reshape(y,[-1,text_num,source_long]),2)

    correct_pred=tf.equal(y_pred,y_true)
    acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))

    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i=0
        while True:
            x1, y1 = get_next_batch(64)
            _acc,_loss,w = sess.run([acc,loss,output], feed_dict={x: x1, y: y1, keep_prob: 0.7})
            i+=1
            print(("损失值：{0}   准确率：{1:>6.1%}   第{2}次").format(_loss,_acc,i))
            if _acc>0.85:
                saver.save(sess,'./model/cc.ckpt',globel_step=i)
            if (i%20==0):
                x1, y1 = get_next_batch(64)
                _acc = sess.run(acc, feed_dict={x: x1, y: y1, keep_prob: 0.7})
                print(("准确率：{0:>6.1%}").format(_acc))



def text2batch(text_source):
    labels = np.zeros(text_num * source_long)
    text1=text_source[0:1]
    for text_f in alphabet:
        if text_f==text1:
            index=alphabet.index(text_f)
            labels[index]=1.0

    text2 = text_source[1:2]
    for text_f in alphabet:
        if text_f == text2:
            index = alphabet.index(text_f)
            labels[index+26] = 1.0

    text3 = text_source[2:3]
    for text_f in alphabet:
        if text_f == text3:
            index = alphabet.index(text_f)
            labels[index + 52] = 1.0

    text4 = text_source[3:4]
    for text_f in alphabet:
        if text_f == text4:
            index = alphabet.index(text_f)
            labels[index + 78] = 1.0
    return labels

if __name__=='__main__':
    #captcha_img,captcha_text=gen_captcha_text_img(alphabet,4)
    #captcha_img = color.rgb2gray(captcha_img)
    #plt.imshow(captcha_img)
    #plt.show()#将生成的图片进行展示

    img_x=160
    img_y=60
    text_num=4
    source_long=len(alphabet)
    batch_size=64
    x=tf.placeholder(tf.float32,[None,img_x*img_y])
    y=tf.placeholder(tf.float32,[None,text_num*source_long])
    x_batch=np.zeros([batch_size,img_x*img_y])
    y_batch=np.zeros([batch_size,text_num*source_long])
    keep_prob=tf.placeholder(tf.float32)
    train_model()


