import tensorflow as tf
import numpy as np
print("\n先测试一维张量\n")
t=np.random.randint(1,10,5)
g1=tf.gather(t,[2,1,4])
sess=tf.Session()
print(t)
print(sess.run(g1))
print("\n再测试二维张量\n")
t=np.random.randint(1,10,[4,5])
g2=tf.gather(t,[1,2,2],axis=0)
g3=tf.gather(t,[1,2,2],axis=1)
print(t)
print(sess.run(g2))
print(sess.run(g3))
print(sess.graph_def)





"""
indices = tf.constant([0, 1, 1])
x = tf.constant([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9]])

result = tf.gather(x, indices)

with tf.Session() as sess:
    out = sess.run(result)
    print(out)
    print(tf.get_default_graph().as_graph_def())
"""

##########   GraphDef ############
node {
  name: "in_data"
  op: "Placeholder"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 4
          }
        }
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 4
        }
      }
    }
  }
}
node {
  name: "indices"
  op: "Placeholder"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1
          }
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "shape"
    value {
      shape {
        dim {
          size: 1
        }
        dim {
          size: 2
        }
        dim {
          size: 2
        }
      }
    }
  }
}
node {
  name: "GatherV2/axis"
  op: "Const"
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
        }
      }
    }
  }
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "GatherV2"
  op: "GatherV2"
  input: "in_data"
  input: "indices"
  input: "GatherV2/axis"
  attr {
    key: "Taxis"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tindices"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tparams"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "_output_shapes"
    value {
      list {
        shape {
          dim {
            size: 1
          }
          dim {
            size: 2
          }
          dim {
            size: 2
          }
        }
      }
    }
  }
}
library {
}