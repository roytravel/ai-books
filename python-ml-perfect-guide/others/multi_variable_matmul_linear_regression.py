import tensorflow.compat.v1 as tf

# check eager executing is enabled
# print (tf.executing_eagerly())

tf.compat.v1.disable_eager_execution()


x_data = [[73., 80., 75.], [93., 88., 93.],
          [89., 91., 90.], [96., 98., 100.], [73., 66., 70.]]
y_data = [[152.], [185.], [180.], [196.], [142.]]


# placeholder 전달 파라미터
'''  placeholder = allocate other tensor.
placeholder (
  dtype, // 필수
  shape=None, // 입력 데이터의 형태 (상수 or 다차원 배열)
  name=None // placeholder의 이름 부여하는 
)
'''
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Hypothesis
W = tf.Variable(tf.random_normal([3, 1]), name='weight')

# Simplied cost/loss function
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
  cost_val, hy_val, _ = sess.run(
      [cost, hypothesis, train], feed_dict = {X: x_data, Y: y_data})
  
  if step % 10 == 0:
    print (step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

