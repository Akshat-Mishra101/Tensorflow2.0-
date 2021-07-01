import tensorflow as tf
#creating tensors
Integer_tensor=tf.Variable(324,tf.int16)#this is an integer tensor

String_tensor=tf.Variable("String",tf.string)#this is a String tensor

float_tensor=tf.Variable(3.567,tf.float64)#this is a float tensor

#These Tensors are scalars since they only have a single value

#scalars are rank zero tensors
#if a tensor has two values then it is a tensor of second rank
#number of dimensions = rank


rank2_tensor =tf.Variable([["hey","ther","there"],["hey","where","there"],["wherever","ther","i"],["wherever","i","want"]],tf.string)#4x3 tensor

#tensorflow uses numpy for a lot of things

print(tf.rank(rank2_tensor))#this prints the rank or the degree of the tensor (the total number of dimensions)
print(rank2_tensor.shape)#shape = the dimensions ex 4x3

rank3_tensor =tf.Variable([[["hey","there"],["hey","there"]],[["hey","there"],["hey","there"]],[["hey","there"],["hey","there"]],[["hey","there"],["hey","there"]]],tf.string)#4x3 tensor

print(tf.rank(rank3_tensor))
print(rank3_tensor.shape)

onesies=tf.ones([1,2,3])#creates an identity tensor with the given dimensions

print(onesies)

#Reshaping tensors
#we can reshape tensors using the function reshape

reshaped_onesies=tf.reshape(onesies,[3,1,2])
print(reshaped_onesies)

#The Other Reshaping
reshaped_2sie=tf.reshape(onesies,[3,-1])
print(reshaped_2sie)
