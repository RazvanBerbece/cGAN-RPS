??#
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8?? 
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	d?@*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	d?@*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?@*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?@*
dtype0
?
embedding/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*%
shared_nameembedding/embeddings
}
(embedding/embeddings/Read/ReadVariableOpReadVariableOpembedding/embeddings*
_output_shapes

:*
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
?
conv_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_1/kernel
?
+conv_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel*(
_output_shapes
:??*
dtype0
m

bn_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_1/gamma
f
bn_1/gamma/Read/ReadVariableOpReadVariableOp
bn_1/gamma*
_output_shapes	
:?*
dtype0
k
	bn_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_1/beta
d
bn_1/beta/Read/ReadVariableOpReadVariableOp	bn_1/beta*
_output_shapes	
:?*
dtype0
y
bn_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_1/moving_mean
r
$bn_1/moving_mean/Read/ReadVariableOpReadVariableOpbn_1/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_1/moving_variance
z
(bn_1/moving_variance/Read/ReadVariableOpReadVariableOpbn_1/moving_variance*
_output_shapes	
:?*
dtype0
?
conv_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_2/kernel
?
+conv_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel*(
_output_shapes
:??*
dtype0
m

bn_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_2/gamma
f
bn_2/gamma/Read/ReadVariableOpReadVariableOp
bn_2/gamma*
_output_shapes	
:?*
dtype0
k
	bn_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_2/beta
d
bn_2/beta/Read/ReadVariableOpReadVariableOp	bn_2/beta*
_output_shapes	
:?*
dtype0
y
bn_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_2/moving_mean
r
$bn_2/moving_mean/Read/ReadVariableOpReadVariableOpbn_2/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_2/moving_variance
z
(bn_2/moving_variance/Read/ReadVariableOpReadVariableOpbn_2/moving_variance*
_output_shapes	
:?*
dtype0
?
conv_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_3/kernel
?
+conv_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_3/kernel*(
_output_shapes
:??*
dtype0
m

bn_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_3/gamma
f
bn_3/gamma/Read/ReadVariableOpReadVariableOp
bn_3/gamma*
_output_shapes	
:?*
dtype0
k
	bn_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_3/beta
d
bn_3/beta/Read/ReadVariableOpReadVariableOp	bn_3/beta*
_output_shapes	
:?*
dtype0
y
bn_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_3/moving_mean
r
$bn_3/moving_mean/Read/ReadVariableOpReadVariableOpbn_3/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_3/moving_variance
z
(bn_3/moving_variance/Read/ReadVariableOpReadVariableOpbn_3/moving_variance*
_output_shapes	
:?*
dtype0
?
conv_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_4/kernel
?
+conv_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_4/kernel*(
_output_shapes
:??*
dtype0
m

bn_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_4/gamma
f
bn_4/gamma/Read/ReadVariableOpReadVariableOp
bn_4/gamma*
_output_shapes	
:?*
dtype0
k
	bn_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_4/beta
d
bn_4/beta/Read/ReadVariableOpReadVariableOp	bn_4/beta*
_output_shapes	
:?*
dtype0
y
bn_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_4/moving_mean
r
$bn_4/moving_mean/Read/ReadVariableOpReadVariableOpbn_4/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_4/moving_variance
z
(bn_4/moving_variance/Read/ReadVariableOpReadVariableOpbn_4/moving_variance*
_output_shapes	
:?*
dtype0
?
conv_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv_transpose_5/kernel
?
+conv_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_5/kernel*'
_output_shapes
:?*
dtype0

NoOpNoOp
?Y
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?X
value?XB?X B?X
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories
?

kernel
 bias
#!_self_saveable_object_factories
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?
&
embeddings
#'_self_saveable_object_factories
(trainable_variables
)	variables
*regularization_losses
+	keras_api
w
#,_self_saveable_object_factories
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?

1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5	variables
6regularization_losses
7	keras_api
w
#8_self_saveable_object_factories
9trainable_variables
:	variables
;regularization_losses
<	keras_api
w
#=_self_saveable_object_factories
>trainable_variables
?	variables
@regularization_losses
A	keras_api
w
#B_self_saveable_object_factories
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
?

Gkernel
#H_self_saveable_object_factories
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
#R_self_saveable_object_factories
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
w
#W_self_saveable_object_factories
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
?

\kernel
#]_self_saveable_object_factories
^trainable_variables
_	variables
`regularization_losses
a	keras_api
?
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance
#g_self_saveable_object_factories
htrainable_variables
i	variables
jregularization_losses
k	keras_api
w
#l_self_saveable_object_factories
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?

qkernel
#r_self_saveable_object_factories
strainable_variables
t	variables
uregularization_losses
v	keras_api
?
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
#|_self_saveable_object_factories
}trainable_variables
~	variables
regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
|
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?
?kernel
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
 
?
0
 1
&2
13
24
G5
N6
O7
\8
c9
d10
q11
x12
y13
?14
?15
?16
?17
 
?
0
 1
&2
13
24
G5
N6
O7
P8
Q9
\10
c11
d12
e13
f14
q15
x16
y17
z18
{19
?20
?21
?22
?23
?24
?25
?
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?metrics
	variables
 
 
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
 
?
"trainable_variables
?layer_metrics
#	variables
 ?layer_regularization_losses
$regularization_losses
?layers
?metrics
?non_trainable_variables
db
VARIABLE_VALUEembedding/embeddings:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
 
?
(trainable_variables
?layer_metrics
)	variables
 ?layer_regularization_losses
*regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
-trainable_variables
?layer_metrics
.	variables
 ?layer_regularization_losses
/regularization_losses
?layers
?metrics
?non_trainable_variables
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

10
21

10
21
 
?
4trainable_variables
?layer_metrics
5	variables
 ?layer_regularization_losses
6regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
9trainable_variables
?layer_metrics
:	variables
 ?layer_regularization_losses
;regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
>trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
@regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
Ctrainable_variables
?layer_metrics
D	variables
 ?layer_regularization_losses
Eregularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

G0

G0
 
?
Itrainable_variables
?layer_metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
?layers
?metrics
?non_trainable_variables
 
US
VARIABLE_VALUE
bn_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

N0
O1

N0
O1
P2
Q3
 
?
Strainable_variables
?layer_metrics
T	variables
 ?layer_regularization_losses
Uregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
Xtrainable_variables
?layer_metrics
Y	variables
 ?layer_regularization_losses
Zregularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_2/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

\0

\0
 
?
^trainable_variables
?layer_metrics
_	variables
 ?layer_regularization_losses
`regularization_losses
?layers
?metrics
?non_trainable_variables
 
US
VARIABLE_VALUE
bn_2/gamma5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_2/beta4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_2/moving_mean;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_2/moving_variance?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

c0
d1

c0
d1
e2
f3
 
?
htrainable_variables
?layer_metrics
i	variables
 ?layer_regularization_losses
jregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
mtrainable_variables
?layer_metrics
n	variables
 ?layer_regularization_losses
oregularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_3/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

q0

q0
 
?
strainable_variables
?layer_metrics
t	variables
 ?layer_regularization_losses
uregularization_losses
?layers
?metrics
?non_trainable_variables
 
US
VARIABLE_VALUE
bn_3/gamma5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUE	bn_3/beta4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEbn_3/moving_mean;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEbn_3/moving_variance?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

x0
y1
z2
{3
 
?
}trainable_variables
?layer_metrics
~	variables
 ?layer_regularization_losses
regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_4/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

?0

?0
 
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
 
VT
VARIABLE_VALUE
bn_4/gamma6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE	bn_4/beta5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbn_4/moving_mean<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEbn_4/moving_variance@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
 
?0
?1
?2
?3
 
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
db
VARIABLE_VALUEconv_transpose_5/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

?0

?0
 
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
:
P0
Q1
e2
f3
z4
{5
?6
?7
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

P0
Q1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

e0
f1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

z0
{1
 
 
 
 
 
 
 
 
 
 
 
 
 
 

?0
?1
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
z
serving_default_input_2Placeholder*'
_output_shapes
:?????????d*
dtype0*
shape:?????????d
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2embedding/embeddingsdense_1/kerneldense_1/biasdense/kernel
dense/biasconv_transpose_1/kernel
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv_transpose_2/kernel
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_varianceconv_transpose_3/kernel
bn_3/gamma	bn_3/betabn_3/moving_meanbn_3/moving_varianceconv_transpose_4/kernel
bn_4/gamma	bn_4/betabn_4/moving_meanbn_4/moving_varianceconv_transpose_5/kernel*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2628511
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
z
StaticRegexFullMatchStaticRegexFullMatchsaver_filename"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*
\
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part
a
Const_2Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
h
SelectSelectStaticRegexFullMatchConst_1Const_2"/device:CPU:**
T0*
_output_shapes
: 
`

StringJoin
StringJoinsaver_filenameSelect"/device:CPU:**
N*
_output_shapes
: 
L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :
f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
x
ShardedFilenameShardedFilename
StringJoinShardedFilename/shard
num_shards"/device:CPU:0*
_output_shapes
: 
?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp(embedding/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp+conv_transpose_1/kernel/Read/ReadVariableOpbn_1/gamma/Read/ReadVariableOpbn_1/beta/Read/ReadVariableOp$bn_1/moving_mean/Read/ReadVariableOp(bn_1/moving_variance/Read/ReadVariableOp+conv_transpose_2/kernel/Read/ReadVariableOpbn_2/gamma/Read/ReadVariableOpbn_2/beta/Read/ReadVariableOp$bn_2/moving_mean/Read/ReadVariableOp(bn_2/moving_variance/Read/ReadVariableOp+conv_transpose_3/kernel/Read/ReadVariableOpbn_3/gamma/Read/ReadVariableOpbn_3/beta/Read/ReadVariableOp$bn_3/moving_mean/Read/ReadVariableOp(bn_3/moving_variance/Read/ReadVariableOp+conv_transpose_4/kernel/Read/ReadVariableOpbn_4/gamma/Read/ReadVariableOpbn_4/beta/Read/ReadVariableOp$bn_4/moving_mean/Read/ReadVariableOp(bn_4/moving_variance/Read/ReadVariableOp+conv_transpose_5/kernel/Read/ReadVariableOpConst"/device:CPU:0*)
dtypes
2
?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
o
MergeV2CheckpointsMergeV2Checkpoints&MergeV2Checkpoints/checkpoint_prefixessaver_filename"/device:CPU:0
i
IdentityIdentitysaver_filename^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-10/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-10/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-10/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*I
value@B>B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*?
_output_shapesn
l:::::::::::::::::::::::::::*)
dtypes
2
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOpAssignVariableOpdense_1/kernel
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_1AssignVariableOpdense_1/bias
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_2AssignVariableOpembedding/embeddings
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_3AssignVariableOpdense/kernel
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_4AssignVariableOp
dense/bias
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_5AssignVariableOpconv_transpose_1/kernel
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_6AssignVariableOp
bn_1/gamma
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_7AssignVariableOp	bn_1/beta
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_8AssignVariableOpbn_1/moving_mean
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
e
AssignVariableOp_9AssignVariableOpbn_1/moving_varianceIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_10AssignVariableOpconv_transpose_2/kernelIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_11AssignVariableOp
bn_2/gammaIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_12AssignVariableOp	bn_2/betaIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_13AssignVariableOpbn_2/moving_meanIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_14AssignVariableOpbn_2/moving_varianceIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_15AssignVariableOpconv_transpose_3/kernelIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_16AssignVariableOp
bn_3/gammaIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_17AssignVariableOp	bn_3/betaIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_18AssignVariableOpbn_3/moving_meanIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_19AssignVariableOpbn_3/moving_varianceIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_20AssignVariableOpconv_transpose_4/kernelIdentity_21"/device:CPU:0*
dtype0
W
Identity_22IdentityRestoreV2:21"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_21AssignVariableOp
bn_4/gammaIdentity_22"/device:CPU:0*
dtype0
W
Identity_23IdentityRestoreV2:22"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_22AssignVariableOp	bn_4/betaIdentity_23"/device:CPU:0*
dtype0
W
Identity_24IdentityRestoreV2:23"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_23AssignVariableOpbn_4/moving_meanIdentity_24"/device:CPU:0*
dtype0
W
Identity_25IdentityRestoreV2:24"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_24AssignVariableOpbn_4/moving_varianceIdentity_25"/device:CPU:0*
dtype0
W
Identity_26IdentityRestoreV2:25"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_25AssignVariableOpconv_transpose_5/kernelIdentity_26"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_27Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??
?
^
B__inference_re_lu_layer_call_and_return_conditional_losses_2629389

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????@2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
&__inference_bn_1_layer_call_fn_2629673

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_conv_transpose_4_layer_call_fn_2630331

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630235

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
t
H__inference_concatenate_layer_call_and_return_conditional_losses_2629519
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:??????????:?????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
&__inference_bn_4_layer_call_fn_2630453

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630381

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
2__inference_conv_transpose_1_layer_call_fn_2629569

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_relu_2_layer_call_and_return_conditional_losses_2630027

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630253

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
D
(__inference_relu_1_layer_call_fn_2629768

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_bn_4_layer_call_fn_2630435

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629763

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_relu_2_layer_call_fn_2630022

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_relu_3_layer_call_and_return_conditional_losses_2630281

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630471

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
B__inference_model_layer_call_and_return_conditional_losses_2628451
input_1
input_24
"embedding_embedding_lookup_2628249:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?bn_4/AssignNewValue?bn_4/AssignNewValue_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookupr
embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2628249embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2628249*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2628249*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_2%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_1/FusedBatchNormV3?
bn_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"bn_1/FusedBatchNormV3:batch_mean:0%^bn_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_1/AssignNewValue?
bn_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&bn_1/FusedBatchNormV3:batch_variance:0'^bn_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_1/AssignNewValue_1x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_2/FusedBatchNormV3?
bn_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"bn_2/FusedBatchNormV3:batch_mean:0%^bn_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_2/AssignNewValue?
bn_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&bn_2/FusedBatchNormV3:batch_variance:0'^bn_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_2/AssignNewValue_1x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_3/FusedBatchNormV3?
bn_3/AssignNewValueAssignVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource"bn_3/FusedBatchNormV3:batch_mean:0%^bn_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_3/AssignNewValue?
bn_3/AssignNewValue_1AssignVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource&bn_3/FusedBatchNormV3:batch_variance:0'^bn_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_3/AssignNewValue_1x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_4/FusedBatchNormV3?
bn_4/AssignNewValueAssignVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource"bn_4/FusedBatchNormV3:batch_mean:0%^bn_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_4/AssignNewValue?
bn_4/AssignNewValue_1AssignVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource&bn_4/FusedBatchNormV3:batch_variance:0'^bn_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_4/AssignNewValue_1x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?	
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1^bn_4/AssignNewValue^bn_4/AssignNewValue_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
bn_1/AssignNewValuebn_1/AssignNewValue2.
bn_1/AssignNewValue_1bn_1/AssignNewValue_12L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12*
bn_2/AssignNewValuebn_2/AssignNewValue2.
bn_2/AssignNewValue_1bn_2/AssignNewValue_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12*
bn_3/AssignNewValuebn_3/AssignNewValue2.
bn_3/AssignNewValue_1bn_3/AssignNewValue_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12*
bn_4/AssignNewValuebn_4/AssignNewValue2.
bn_4/AssignNewValue_1bn_4/AssignNewValue_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629709

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
'__inference_model_layer_call_fn_2626269
input_1
input_24
"embedding_embedding_lookup_2626067:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookupr
embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2626067embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2626067*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2626067*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_2%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_4/FusedBatchNormV3x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629745

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
'__inference_dense_layer_call_fn_2629419

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_bn_4_layer_call_fn_2630417

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
Y
-__inference_concatenate_layer_call_fn_2629512
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:??????????:?????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:YU
/
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
&__inference_bn_3_layer_call_fn_2630163

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630525

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
G
+__inference_reshape_1_layer_call_fn_2629463

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?

?
F__inference_embedding_layer_call_and_return_conditional_losses_2629379

inputs*
embedding_lookup_2629373:
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_2629373Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/2629373*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/2629373*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630271

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
&__inference_bn_1_layer_call_fn_2629637

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?!
?
2__inference_conv_transpose_1_layer_call_fn_2629549

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_relu_1_layer_call_and_return_conditional_losses_2629773

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:??????????2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
2__inference_conv_transpose_2_layer_call_fn_2629803

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_bn_1_layer_call_fn_2629691

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630107

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_relu_3_layer_call_fn_2630276

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????  ?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????  ?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????  ?:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?

?
+__inference_embedding_layer_call_fn_2629369

inputs*
embedding_lookup_2629363:
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_2629363Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/2629363*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/2629363*+
_output_shapes
:?????????2
embedding_lookup/Identity?
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2
embedding_lookup/Identity_1?
IdentityIdentity$embedding_lookup/Identity_1:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identitya
NoOpNoOp^embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:?????????: 2$
embedding_lookupembedding_lookup:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629727

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_bn_2_layer_call_fn_2629891

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
'__inference_model_layer_call_fn_2628925
inputs_0
inputs_14
"embedding_embedding_lookup_2628723:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?bn_4/AssignNewValue?bn_4/AssignNewValue_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookups
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2628723embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2628723*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2628723*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_1/FusedBatchNormV3?
bn_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"bn_1/FusedBatchNormV3:batch_mean:0%^bn_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_1/AssignNewValue?
bn_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&bn_1/FusedBatchNormV3:batch_variance:0'^bn_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_1/AssignNewValue_1x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_2/FusedBatchNormV3?
bn_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"bn_2/FusedBatchNormV3:batch_mean:0%^bn_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_2/AssignNewValue?
bn_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&bn_2/FusedBatchNormV3:batch_variance:0'^bn_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_2/AssignNewValue_1x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_3/FusedBatchNormV3?
bn_3/AssignNewValueAssignVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource"bn_3/FusedBatchNormV3:batch_mean:0%^bn_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_3/AssignNewValue?
bn_3/AssignNewValue_1AssignVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource&bn_3/FusedBatchNormV3:batch_variance:0'^bn_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_3/AssignNewValue_1x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_4/FusedBatchNormV3?
bn_4/AssignNewValueAssignVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource"bn_4/FusedBatchNormV3:batch_mean:0%^bn_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_4/AssignNewValue?
bn_4/AssignNewValue_1AssignVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource&bn_4/FusedBatchNormV3:batch_variance:0'^bn_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_4/AssignNewValue_1x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?	
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1^bn_4/AssignNewValue^bn_4/AssignNewValue_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
bn_1/AssignNewValuebn_1/AssignNewValue2.
bn_1/AssignNewValue_1bn_1/AssignNewValue_12L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12*
bn_2/AssignNewValuebn_2/AssignNewValue2.
bn_2/AssignNewValue_1bn_2/AssignNewValue_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12*
bn_3/AssignNewValuebn_3/AssignNewValue2.
bn_3/AssignNewValue_1bn_3/AssignNewValue_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12*
bn_4/AssignNewValuebn_4/AssignNewValue2.
bn_4/AssignNewValue_1bn_4/AssignNewValue_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
?
&__inference_bn_1_layer_call_fn_2629655

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629981

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629873

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630217

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629963

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_conv_transpose_5_layer_call_fn_2630587

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transposek
TanhTanhconv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?!
?
2__inference_conv_transpose_4_layer_call_fn_2630311

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
ی
?
B__inference_model_layer_call_and_return_conditional_losses_2629132
inputs_0
inputs_14
"embedding_embedding_lookup_2628930:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookups
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2628930embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2628930*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2628930*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_4/FusedBatchNormV3x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
?
2__inference_conv_transpose_3_layer_call_fn_2630077

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
? 
?
B__inference_dense_layer_call_and_return_conditional_losses_2629449

inputs3
!tensordot_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:*
dtype02
Tensordot/ReadVariableOpj
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/axesq
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
Tensordot/freeX
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
:2
Tensordot/Shapet
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2/axis?
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2x
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/GatherV2_1/axis?
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
Tensordot/GatherV2_1l
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const?
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: 2
Tensordot/Prodp
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
Tensordot/Const_1?
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
Tensordot/Prod_1p
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat/axis?
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat?
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
Tensordot/stack?
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
Tensordot/transpose?
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
Tensordot/Reshape?
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
Tensordot/MatMulp
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
Tensordot/Const_2t
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
Tensordot/concat_1/axis?
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
Tensordot/concat_1?
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2	
BiasAddo
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:?????????2

Identity?
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2630145

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2630199

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630507

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????@@?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2628511
input_1
input_2
unknown:
	unknown_0:	d?@
	unknown_1:	?@
	unknown_2:
	unknown_3:%
	unknown_4:??
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?&

unknown_14:??

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:	?&

unknown_19:??

unknown_20:	?

unknown_21:	?

unknown_22:	?

unknown_23:	?%

unknown_24:?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*<
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_26248632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
??
?
'__inference_model_layer_call_fn_2628037
input_1
input_24
"embedding_embedding_lookup_2627835:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?bn_4/AssignNewValue?bn_4/AssignNewValue_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookupr
embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2627835embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2627835*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2627835*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_2%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_1/FusedBatchNormV3?
bn_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"bn_1/FusedBatchNormV3:batch_mean:0%^bn_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_1/AssignNewValue?
bn_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&bn_1/FusedBatchNormV3:batch_variance:0'^bn_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_1/AssignNewValue_1x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_2/FusedBatchNormV3?
bn_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"bn_2/FusedBatchNormV3:batch_mean:0%^bn_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_2/AssignNewValue?
bn_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&bn_2/FusedBatchNormV3:batch_variance:0'^bn_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_2/AssignNewValue_1x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_3/FusedBatchNormV3?
bn_3/AssignNewValueAssignVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource"bn_3/FusedBatchNormV3:batch_mean:0%^bn_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_3/AssignNewValue?
bn_3/AssignNewValue_1AssignVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource&bn_3/FusedBatchNormV3:batch_variance:0'^bn_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_3/AssignNewValue_1x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_4/FusedBatchNormV3?
bn_4/AssignNewValueAssignVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource"bn_4/FusedBatchNormV3:batch_mean:0%^bn_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_4/AssignNewValue?
bn_4/AssignNewValue_1AssignVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource&bn_4/FusedBatchNormV3:batch_variance:0'^bn_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_4/AssignNewValue_1x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?	
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1^bn_4/AssignNewValue^bn_4/AssignNewValue_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
bn_1/AssignNewValuebn_1/AssignNewValue2.
bn_1/AssignNewValue_1bn_1/AssignNewValue_12L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12*
bn_2/AssignNewValuebn_2/AssignNewValue2.
bn_2/AssignNewValue_1bn_2/AssignNewValue_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12*
bn_3/AssignNewValuebn_3/AssignNewValue2.
bn_3/AssignNewValue_1bn_3/AssignNewValue_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12*
bn_4/AssignNewValuebn_4/AssignNewValue2.
bn_4/AssignNewValue_1bn_4/AssignNewValue_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
?
?
&__inference_bn_3_layer_call_fn_2630181

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:?????????  ?: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
Ռ
?
B__inference_model_layer_call_and_return_conditional_losses_2628244
input_1
input_24
"embedding_embedding_lookup_2628042:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookupr
embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2628042embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2628042*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2628042*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_2%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_4/FusedBatchNormV3x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
?
C
'__inference_re_lu_layer_call_fn_2629384

inputs
identityO
ReluReluinputs*
T0*(
_output_shapes
:??????????@2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:??????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2630017

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_relu_4_layer_call_fn_2630530

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????@@?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
&__inference_bn_2_layer_call_fn_2629927

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629999

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_relu_4_layer_call_and_return_conditional_losses_2630535

inputs
identityW
ReluReluinputs*
T0*0
_output_shapes
:?????????@@?2
Reluo
IdentityIdentityRelu:activations:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:?????????@@?:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630489

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
b
F__inference_reshape_1_layer_call_and_return_conditional_losses_2629477

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????@:P L
(
_output_shapes
:??????????@
 
_user_specified_nameinputs
?!
?
2__inference_conv_transpose_3_layer_call_fn_2630057

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
&__inference_bn_4_layer_call_fn_2630399

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
FusedBatchNormV3?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
2__inference_conv_transpose_2_layer_call_fn_2629823

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_bn_2_layer_call_fn_2629945

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1x
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*0
_output_shapes
:??????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?!
?
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630361

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?"
?
2__inference_conv_transpose_5_layer_call_fn_2630566

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose{
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?"
?
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630618

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose{
TanhTanhconv2d_transpose:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh}
IdentityIdentityTanh:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
`
D__inference_reshape_layer_call_and_return_conditional_losses_2629505

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
D__inference_dense_1_layer_call_and_return_conditional_losses_2629359

inputs1
matmul_readvariableop_resource:	d?@.
biasadd_readvariableop_resource:	?@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?
?
&__inference_bn_2_layer_call_fn_2629909

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1u
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp{
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype02
ReadVariableOp_1?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02!
FusedBatchNormV3/ReadVariableOp?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
FusedBatchNormV3?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
AssignNewValue?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignNewValue_1?
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
B__inference_model_layer_call_and_return_conditional_losses_2629339
inputs_0
inputs_14
"embedding_embedding_lookup_2629137:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?bn_4/AssignNewValue?bn_4/AssignNewValue_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookups
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2629137embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2629137*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2629137*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_1/FusedBatchNormV3?
bn_1/AssignNewValueAssignVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource"bn_1/FusedBatchNormV3:batch_mean:0%^bn_1/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_1/AssignNewValue?
bn_1/AssignNewValue_1AssignVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource&bn_1/FusedBatchNormV3:batch_variance:0'^bn_1/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_1/AssignNewValue_1x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_2/FusedBatchNormV3?
bn_2/AssignNewValueAssignVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource"bn_2/FusedBatchNormV3:batch_mean:0%^bn_2/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_2/AssignNewValue?
bn_2/AssignNewValue_1AssignVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource&bn_2/FusedBatchNormV3:batch_variance:0'^bn_2/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_2/AssignNewValue_1x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_3/FusedBatchNormV3?
bn_3/AssignNewValueAssignVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource"bn_3/FusedBatchNormV3:batch_mean:0%^bn_3/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_3/AssignNewValue?
bn_3/AssignNewValue_1AssignVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource&bn_3/FusedBatchNormV3:batch_variance:0'^bn_3/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_3/AssignNewValue_1x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
exponential_avg_factor%fff?2
bn_4/FusedBatchNormV3?
bn_4/AssignNewValueAssignVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource"bn_4/FusedBatchNormV3:batch_mean:0%^bn_4/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype02
bn_4/AssignNewValue?
bn_4/AssignNewValue_1AssignVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource&bn_4/FusedBatchNormV3:batch_variance:0'^bn_4/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype02
bn_4/AssignNewValue_1x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?	
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1^bn_4/AssignNewValue^bn_4/AssignNewValue_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2*
bn_1/AssignNewValuebn_1/AssignNewValue2.
bn_1/AssignNewValue_1bn_1/AssignNewValue_12L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12*
bn_2/AssignNewValuebn_2/AssignNewValue2.
bn_2/AssignNewValue_1bn_2/AssignNewValue_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12*
bn_3/AssignNewValuebn_3/AssignNewValue2.
bn_3/AssignNewValue_1bn_3/AssignNewValue_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12*
bn_4/AssignNewValuebn_4/AssignNewValue2.
bn_4/AssignNewValue_1bn_4/AssignNewValue_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
?
?
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630639

inputsC
(conv2d_transpose_readvariableop_resource:?
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceU
stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/1U
stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/2T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
conv2d_transposek
TanhTanhconv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
Tanhm
IdentityIdentityTanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?!
?
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629599

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629619

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
)__inference_dense_1_layer_call_fn_2629349

inputs1
matmul_readvariableop_resource:	d?@.
biasadd_readvariableop_resource:	?@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2	
BiasAddl
IdentityIdentityBiasAdd:output:0^NoOp*
T0*(
_output_shapes
:??????????@2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????d: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????d
 
_user_specified_nameinputs
?!
?
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629853

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*C
_input_shapes2
0:,????????????????????????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630127

inputsD
(conv2d_transpose_readvariableop_resource:??
identity??conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliceT
stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/1T
stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/2U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0stack/1:output:0stack/2:output:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSlicestack:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv2d_transpose}
IdentityIdentityconv2d_transpose:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityp
NoOpNoOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
'__inference_model_layer_call_fn_2628718
inputs_0
inputs_14
"embedding_embedding_lookup_2628516:9
&dense_1_matmul_readvariableop_resource:	d?@6
'dense_1_biasadd_readvariableop_resource:	?@9
'dense_tensordot_readvariableop_resource:3
%dense_biasadd_readvariableop_resource:U
9conv_transpose_1_conv2d_transpose_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_2_conv2d_transpose_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_3_conv2d_transpose_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?U
9conv_transpose_4_conv2d_transpose_readvariableop_resource:??+
bn_4_readvariableop_resource:	?-
bn_4_readvariableop_1_resource:	?<
-bn_4_fusedbatchnormv3_readvariableop_resource:	?>
/bn_4_fusedbatchnormv3_readvariableop_1_resource:	?T
9conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?$bn_4/FusedBatchNormV3/ReadVariableOp?&bn_4/FusedBatchNormV3/ReadVariableOp_1?bn_4/ReadVariableOp?bn_4/ReadVariableOp_1?0conv_transpose_1/conv2d_transpose/ReadVariableOp?0conv_transpose_2/conv2d_transpose/ReadVariableOp?0conv_transpose_3/conv2d_transpose/ReadVariableOp?0conv_transpose_4/conv2d_transpose/ReadVariableOp?0conv_transpose_5/conv2d_transpose/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/Tensordot/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?embedding/embedding_lookups
embedding/CastCastinputs_0*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding/Cast?
embedding/embedding_lookupResourceGather"embedding_embedding_lookup_2628516embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*5
_class+
)'loc:@embedding/embedding_lookup/2628516*+
_output_shapes
:?????????*
dtype02
embedding/embedding_lookup?
#embedding/embedding_lookup/IdentityIdentity#embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*5
_class+
)'loc:@embedding/embedding_lookup/2628516*+
_output_shapes
:?????????2%
#embedding/embedding_lookup/Identity?
%embedding/embedding_lookup/Identity_1Identity,embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2'
%embedding/embedding_lookup/Identity_1?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
dense_1/BiasAdd?
dense/Tensordot/ReadVariableOpReadVariableOp'dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02 
dense/Tensordot/ReadVariableOpv
dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/axes}
dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense/Tensordot/free?
dense/Tensordot/ShapeShape.embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense/Tensordot/Shape?
dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/GatherV2/axis?
dense/Tensordot/GatherV2GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/free:output:0&dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2?
dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense/Tensordot/GatherV2_1/axis?
dense/Tensordot/GatherV2_1GatherV2dense/Tensordot/Shape:output:0dense/Tensordot/axes:output:0(dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense/Tensordot/GatherV2_1x
dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const?
dense/Tensordot/ProdProd!dense/Tensordot/GatherV2:output:0dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod|
dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense/Tensordot/Const_1?
dense/Tensordot/Prod_1Prod#dense/Tensordot/GatherV2_1:output:0 dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense/Tensordot/Prod_1|
dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat/axis?
dense/Tensordot/concatConcatV2dense/Tensordot/free:output:0dense/Tensordot/axes:output:0$dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat?
dense/Tensordot/stackPackdense/Tensordot/Prod:output:0dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/stack?
dense/Tensordot/transpose	Transpose.embedding/embedding_lookup/Identity_1:output:0dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot/transpose?
dense/Tensordot/ReshapeReshapedense/Tensordot/transpose:y:0dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense/Tensordot/Reshape?
dense/Tensordot/MatMulMatMul dense/Tensordot/Reshape:output:0&dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/Tensordot/MatMul|
dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
dense/Tensordot/Const_2?
dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense/Tensordot/concat_1/axis?
dense/Tensordot/concat_1ConcatV2!dense/Tensordot/GatherV2:output:0 dense/Tensordot/Const_2:output:0&dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense/Tensordot/concat_1?
dense/TensordotReshape dense/Tensordot/MatMul:product:0!dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
dense/Tensordot?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/Tensordot:output:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
dense/BiasAddm

re_lu/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2

re_lu/Reluj
reshape_1/ShapeShapere_lu/Relu:activations:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapere_lu/Relu:activations:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshaped
reshape/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense/BiasAdd:output:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
reshape/Reshapet
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis?
concatenate/concatConcatV2reshape_1/Reshape:output:0reshape/Reshape:output:0 concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
concatenate/concat{
conv_transpose_1/ShapeShapeconcatenate/concat:output:0*
T0*
_output_shapes
:2
conv_transpose_1/Shape?
$conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_1/strided_slice/stack?
&conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_1?
&conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_1/strided_slice/stack_2?
conv_transpose_1/strided_sliceStridedSliceconv_transpose_1/Shape:output:0-conv_transpose_1/strided_slice/stack:output:0/conv_transpose_1/strided_slice/stack_1:output:0/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_1/strided_slicev
conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/1v
conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_1/stack/2w
conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_1/stack/3?
conv_transpose_1/stackPack'conv_transpose_1/strided_slice:output:0!conv_transpose_1/stack/1:output:0!conv_transpose_1/stack/2:output:0!conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_1/stack?
&conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_1/strided_slice_1/stack?
(conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_1?
(conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_1/strided_slice_1/stack_2?
 conv_transpose_1/strided_slice_1StridedSliceconv_transpose_1/stack:output:0/conv_transpose_1/strided_slice_1/stack:output:01conv_transpose_1/strided_slice_1/stack_1:output:01conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_1/strided_slice_1?
0conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_1/conv2d_transpose/ReadVariableOp?
!conv_transpose_1/conv2d_transposeConv2DBackpropInputconv_transpose_1/stack:output:08conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_1/conv2d_transpose?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3*conv_transpose_1/conv2d_transpose:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3x
relu_1/ReluRelubn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_1/Reluy
conv_transpose_2/ShapeShaperelu_1/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_2/Shape?
$conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_2/strided_slice/stack?
&conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_1?
&conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_2/strided_slice/stack_2?
conv_transpose_2/strided_sliceStridedSliceconv_transpose_2/Shape:output:0-conv_transpose_2/strided_slice/stack:output:0/conv_transpose_2/strided_slice/stack_1:output:0/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_2/strided_slicev
conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/1v
conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_2/stack/2w
conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_2/stack/3?
conv_transpose_2/stackPack'conv_transpose_2/strided_slice:output:0!conv_transpose_2/stack/1:output:0!conv_transpose_2/stack/2:output:0!conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_2/stack?
&conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_2/strided_slice_1/stack?
(conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_1?
(conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_2/strided_slice_1/stack_2?
 conv_transpose_2/strided_slice_1StridedSliceconv_transpose_2/stack:output:0/conv_transpose_2/strided_slice_1/stack:output:01conv_transpose_2/strided_slice_1/stack_1:output:01conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_2/strided_slice_1?
0conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_2/conv2d_transpose/ReadVariableOp?
!conv_transpose_2/conv2d_transposeConv2DBackpropInputconv_transpose_2/stack:output:08conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2#
!conv_transpose_2/conv2d_transpose?
bn_2/ReadVariableOpReadVariableOpbn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp?
bn_2/ReadVariableOp_1ReadVariableOpbn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_2/ReadVariableOp_1?
$bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_2/FusedBatchNormV3/ReadVariableOp?
&bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_2/FusedBatchNormV3/ReadVariableOp_1?
bn_2/FusedBatchNormV3FusedBatchNormV3*conv_transpose_2/conv2d_transpose:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3x
relu_2/ReluRelubn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
relu_2/Reluy
conv_transpose_3/ShapeShaperelu_2/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_3/Shape?
$conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_3/strided_slice/stack?
&conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_1?
&conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_3/strided_slice/stack_2?
conv_transpose_3/strided_sliceStridedSliceconv_transpose_3/Shape:output:0-conv_transpose_3/strided_slice/stack:output:0/conv_transpose_3/strided_slice/stack_1:output:0/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_3/strided_slicev
conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/1v
conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2
conv_transpose_3/stack/2w
conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_3/stack/3?
conv_transpose_3/stackPack'conv_transpose_3/strided_slice:output:0!conv_transpose_3/stack/1:output:0!conv_transpose_3/stack/2:output:0!conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_3/stack?
&conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_3/strided_slice_1/stack?
(conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_1?
(conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_3/strided_slice_1/stack_2?
 conv_transpose_3/strided_slice_1StridedSliceconv_transpose_3/stack:output:0/conv_transpose_3/strided_slice_1/stack:output:01conv_transpose_3/strided_slice_1/stack_1:output:01conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_3/strided_slice_1?
0conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_3/conv2d_transpose/ReadVariableOp?
!conv_transpose_3/conv2d_transposeConv2DBackpropInputconv_transpose_3/stack:output:08conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2#
!conv_transpose_3/conv2d_transpose?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3*conv_transpose_3/conv2d_transpose:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3x
relu_3/ReluRelubn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
relu_3/Reluy
conv_transpose_4/ShapeShaperelu_3/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_4/Shape?
$conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_4/strided_slice/stack?
&conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_1?
&conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_4/strided_slice/stack_2?
conv_transpose_4/strided_sliceStridedSliceconv_transpose_4/Shape:output:0-conv_transpose_4/strided_slice/stack:output:0/conv_transpose_4/strided_slice/stack_1:output:0/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_4/strided_slicev
conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/1v
conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2
conv_transpose_4/stack/2w
conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_4/stack/3?
conv_transpose_4/stackPack'conv_transpose_4/strided_slice:output:0!conv_transpose_4/stack/1:output:0!conv_transpose_4/stack/2:output:0!conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_4/stack?
&conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_4/strided_slice_1/stack?
(conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_1?
(conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_4/strided_slice_1/stack_2?
 conv_transpose_4/strided_slice_1StridedSliceconv_transpose_4/stack:output:0/conv_transpose_4/strided_slice_1/stack:output:01conv_transpose_4/strided_slice_1/stack_1:output:01conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_4/strided_slice_1?
0conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv_transpose_4/conv2d_transpose/ReadVariableOp?
!conv_transpose_4/conv2d_transposeConv2DBackpropInputconv_transpose_4/stack:output:08conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2#
!conv_transpose_4/conv2d_transpose?
bn_4/ReadVariableOpReadVariableOpbn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp?
bn_4/ReadVariableOp_1ReadVariableOpbn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_4/ReadVariableOp_1?
$bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_4/FusedBatchNormV3/ReadVariableOp?
&bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_4/FusedBatchNormV3/ReadVariableOp_1?
bn_4/FusedBatchNormV3FusedBatchNormV3*conv_transpose_4/conv2d_transpose:output:0bn_4/ReadVariableOp:value:0bn_4/ReadVariableOp_1:value:0,bn_4/FusedBatchNormV3/ReadVariableOp:value:0.bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_4/FusedBatchNormV3x
relu_4/ReluRelubn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
relu_4/Reluy
conv_transpose_5/ShapeShaperelu_4/Relu:activations:0*
T0*
_output_shapes
:2
conv_transpose_5/Shape?
$conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv_transpose_5/strided_slice/stack?
&conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_1?
&conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv_transpose_5/strided_slice/stack_2?
conv_transpose_5/strided_sliceStridedSliceconv_transpose_5/Shape:output:0-conv_transpose_5/strided_slice/stack:output:0/conv_transpose_5/strided_slice/stack_1:output:0/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv_transpose_5/strided_slicew
conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/1w
conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2
conv_transpose_5/stack/2v
conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv_transpose_5/stack/3?
conv_transpose_5/stackPack'conv_transpose_5/strided_slice:output:0!conv_transpose_5/stack/1:output:0!conv_transpose_5/stack/2:output:0!conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv_transpose_5/stack?
&conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv_transpose_5/strided_slice_1/stack?
(conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_1?
(conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv_transpose_5/strided_slice_1/stack_2?
 conv_transpose_5/strided_slice_1StridedSliceconv_transpose_5/stack:output:0/conv_transpose_5/strided_slice_1/stack:output:01conv_transpose_5/strided_slice_1/stack_1:output:01conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv_transpose_5/strided_slice_1?
0conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp9conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype022
0conv_transpose_5/conv2d_transpose/ReadVariableOp?
!conv_transpose_5/conv2d_transposeConv2DBackpropInputconv_transpose_5/stack:output:08conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2#
!conv_transpose_5/conv2d_transpose?
conv_transpose_5/TanhTanh*conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
conv_transpose_5/Tanh~
IdentityIdentityconv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1%^bn_4/FusedBatchNormV3/ReadVariableOp'^bn_4/FusedBatchNormV3/ReadVariableOp_1^bn_4/ReadVariableOp^bn_4/ReadVariableOp_11^conv_transpose_1/conv2d_transpose/ReadVariableOp1^conv_transpose_2/conv2d_transpose/ReadVariableOp1^conv_transpose_3/conv2d_transpose/ReadVariableOp1^conv_transpose_4/conv2d_transpose/ReadVariableOp1^conv_transpose_5/conv2d_transpose/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/Tensordot/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$bn_1/FusedBatchNormV3/ReadVariableOp$bn_1/FusedBatchNormV3/ReadVariableOp2P
&bn_1/FusedBatchNormV3/ReadVariableOp_1&bn_1/FusedBatchNormV3/ReadVariableOp_12*
bn_1/ReadVariableOpbn_1/ReadVariableOp2.
bn_1/ReadVariableOp_1bn_1/ReadVariableOp_12L
$bn_2/FusedBatchNormV3/ReadVariableOp$bn_2/FusedBatchNormV3/ReadVariableOp2P
&bn_2/FusedBatchNormV3/ReadVariableOp_1&bn_2/FusedBatchNormV3/ReadVariableOp_12*
bn_2/ReadVariableOpbn_2/ReadVariableOp2.
bn_2/ReadVariableOp_1bn_2/ReadVariableOp_12L
$bn_3/FusedBatchNormV3/ReadVariableOp$bn_3/FusedBatchNormV3/ReadVariableOp2P
&bn_3/FusedBatchNormV3/ReadVariableOp_1&bn_3/FusedBatchNormV3/ReadVariableOp_12*
bn_3/ReadVariableOpbn_3/ReadVariableOp2.
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12L
$bn_4/FusedBatchNormV3/ReadVariableOp$bn_4/FusedBatchNormV3/ReadVariableOp2P
&bn_4/FusedBatchNormV3/ReadVariableOp_1&bn_4/FusedBatchNormV3/ReadVariableOp_12*
bn_4/ReadVariableOpbn_4/ReadVariableOp2.
bn_4/ReadVariableOp_1bn_4/ReadVariableOp_12d
0conv_transpose_1/conv2d_transpose/ReadVariableOp0conv_transpose_1/conv2d_transpose/ReadVariableOp2d
0conv_transpose_2/conv2d_transpose/ReadVariableOp0conv_transpose_2/conv2d_transpose/ReadVariableOp2d
0conv_transpose_3/conv2d_transpose/ReadVariableOp0conv_transpose_3/conv2d_transpose/ReadVariableOp2d
0conv_transpose_4/conv2d_transpose/ReadVariableOp0conv_transpose_4/conv2d_transpose/ReadVariableOp2d
0conv_transpose_5/conv2d_transpose/ReadVariableOp0conv_transpose_5/conv2d_transpose/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2@
dense/Tensordot/ReadVariableOpdense/Tensordot/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp28
embedding/embedding_lookupembedding/embedding_lookup:Q M
'
_output_shapes
:?????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????d
"
_user_specified_name
inputs/1
??
?
"__inference__wrapped_model_2624863
input_1
input_2:
(model_embedding_embedding_lookup_2624661:?
,model_dense_1_matmul_readvariableop_resource:	d?@<
-model_dense_1_biasadd_readvariableop_resource:	?@?
-model_dense_tensordot_readvariableop_resource:9
+model_dense_biasadd_readvariableop_resource:[
?model_conv_transpose_1_conv2d_transpose_readvariableop_resource:??1
"model_bn_1_readvariableop_resource:	?3
$model_bn_1_readvariableop_1_resource:	?B
3model_bn_1_fusedbatchnormv3_readvariableop_resource:	?D
5model_bn_1_fusedbatchnormv3_readvariableop_1_resource:	?[
?model_conv_transpose_2_conv2d_transpose_readvariableop_resource:??1
"model_bn_2_readvariableop_resource:	?3
$model_bn_2_readvariableop_1_resource:	?B
3model_bn_2_fusedbatchnormv3_readvariableop_resource:	?D
5model_bn_2_fusedbatchnormv3_readvariableop_1_resource:	?[
?model_conv_transpose_3_conv2d_transpose_readvariableop_resource:??1
"model_bn_3_readvariableop_resource:	?3
$model_bn_3_readvariableop_1_resource:	?B
3model_bn_3_fusedbatchnormv3_readvariableop_resource:	?D
5model_bn_3_fusedbatchnormv3_readvariableop_1_resource:	?[
?model_conv_transpose_4_conv2d_transpose_readvariableop_resource:??1
"model_bn_4_readvariableop_resource:	?3
$model_bn_4_readvariableop_1_resource:	?B
3model_bn_4_fusedbatchnormv3_readvariableop_resource:	?D
5model_bn_4_fusedbatchnormv3_readvariableop_1_resource:	?Z
?model_conv_transpose_5_conv2d_transpose_readvariableop_resource:?
identity??*model/bn_1/FusedBatchNormV3/ReadVariableOp?,model/bn_1/FusedBatchNormV3/ReadVariableOp_1?model/bn_1/ReadVariableOp?model/bn_1/ReadVariableOp_1?*model/bn_2/FusedBatchNormV3/ReadVariableOp?,model/bn_2/FusedBatchNormV3/ReadVariableOp_1?model/bn_2/ReadVariableOp?model/bn_2/ReadVariableOp_1?*model/bn_3/FusedBatchNormV3/ReadVariableOp?,model/bn_3/FusedBatchNormV3/ReadVariableOp_1?model/bn_3/ReadVariableOp?model/bn_3/ReadVariableOp_1?*model/bn_4/FusedBatchNormV3/ReadVariableOp?,model/bn_4/FusedBatchNormV3/ReadVariableOp_1?model/bn_4/ReadVariableOp?model/bn_4/ReadVariableOp_1?6model/conv_transpose_1/conv2d_transpose/ReadVariableOp?6model/conv_transpose_2/conv2d_transpose/ReadVariableOp?6model/conv_transpose_3/conv2d_transpose/ReadVariableOp?6model/conv_transpose_4/conv2d_transpose/ReadVariableOp?6model/conv_transpose_5/conv2d_transpose/ReadVariableOp?"model/dense/BiasAdd/ReadVariableOp?$model/dense/Tensordot/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp? model/embedding/embedding_lookup~
model/embedding/CastCastinput_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model/embedding/Cast?
 model/embedding/embedding_lookupResourceGather(model_embedding_embedding_lookup_2624661model/embedding/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*;
_class1
/-loc:@model/embedding/embedding_lookup/2624661*+
_output_shapes
:?????????*
dtype02"
 model/embedding/embedding_lookup?
)model/embedding/embedding_lookup/IdentityIdentity)model/embedding/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*;
_class1
/-loc:@model/embedding/embedding_lookup/2624661*+
_output_shapes
:?????????2+
)model/embedding/embedding_lookup/Identity?
+model/embedding/embedding_lookup/Identity_1Identity2model/embedding/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2-
+model/embedding/embedding_lookup/Identity_1?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	d?@*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMulinput_2+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?@*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????@2
model/dense_1/BiasAdd?
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:*
dtype02&
$model/dense/Tensordot/ReadVariableOp?
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
model/dense/Tensordot/axes?
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
model/dense/Tensordot/free?
model/dense/Tensordot/ShapeShape4model/embedding/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
model/dense/Tensordot/Shape?
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/dense/Tensordot/GatherV2/axis?
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2 
model/dense/Tensordot/GatherV2?
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model/dense/Tensordot/GatherV2_1/axis?
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2"
 model/dense/Tensordot/GatherV2_1?
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
model/dense/Tensordot/Const?
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
model/dense/Tensordot/Prod?
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
model/dense/Tensordot/Const_1?
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
model/dense/Tensordot/Prod_1?
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!model/dense/Tensordot/concat/axis?
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
model/dense/Tensordot/concat?
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
model/dense/Tensordot/stack?
model/dense/Tensordot/transpose	Transpose4model/embedding/embedding_lookup/Identity_1:output:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2!
model/dense/Tensordot/transpose?
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
model/dense/Tensordot/Reshape?
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model/dense/Tensordot/MatMul?
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:2
model/dense/Tensordot/Const_2?
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/dense/Tensordot/concat_1/axis?
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2 
model/dense/Tensordot/concat_1?
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:?????????2
model/dense/Tensordot?
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"model/dense/BiasAdd/ReadVariableOp?
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:?????????2
model/dense/BiasAdd
model/re_lu/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:??????????@2
model/re_lu/Relu|
model/reshape_1/ShapeShapemodel/re_lu/Relu:activations:0*
T0*
_output_shapes
:2
model/reshape_1/Shape?
#model/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#model/reshape_1/strided_slice/stack?
%model/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_1/strided_slice/stack_1?
%model/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%model/reshape_1/strided_slice/stack_2?
model/reshape_1/strided_sliceStridedSlicemodel/reshape_1/Shape:output:0,model/reshape_1/strided_slice/stack:output:0.model/reshape_1/strided_slice/stack_1:output:0.model/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape_1/strided_slice?
model/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_1/Reshape/shape/1?
model/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
model/reshape_1/Reshape/shape/2?
model/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2!
model/reshape_1/Reshape/shape/3?
model/reshape_1/Reshape/shapePack&model/reshape_1/strided_slice:output:0(model/reshape_1/Reshape/shape/1:output:0(model/reshape_1/Reshape/shape/2:output:0(model/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model/reshape_1/Reshape/shape?
model/reshape_1/ReshapeReshapemodel/re_lu/Relu:activations:0&model/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
model/reshape_1/Reshapev
model/reshape/ShapeShapemodel/dense/BiasAdd:output:0*
T0*
_output_shapes
:2
model/reshape/Shape?
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2#
!model/reshape/strided_slice/stack?
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_1?
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2%
#model/reshape/strided_slice/stack_2?
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
model/reshape/strided_slice?
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/1?
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/2?
model/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
model/reshape/Reshape/shape/3?
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0&model/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
model/reshape/Reshape/shape?
model/reshape/ReshapeReshapemodel/dense/BiasAdd:output:0$model/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:?????????2
model/reshape/Reshape?
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
model/concatenate/concat/axis?
model/concatenate/concatConcatV2 model/reshape_1/Reshape:output:0model/reshape/Reshape:output:0&model/concatenate/concat/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
model/concatenate/concat?
model/conv_transpose_1/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
:2
model/conv_transpose_1/Shape?
*model/conv_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_1/strided_slice/stack?
,model/conv_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_1/strided_slice/stack_1?
,model/conv_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_1/strided_slice/stack_2?
$model/conv_transpose_1/strided_sliceStridedSlice%model/conv_transpose_1/Shape:output:03model/conv_transpose_1/strided_slice/stack:output:05model/conv_transpose_1/strided_slice/stack_1:output:05model/conv_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_1/strided_slice?
model/conv_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_1/stack/1?
model/conv_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_1/stack/2?
model/conv_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_1/stack/3?
model/conv_transpose_1/stackPack-model/conv_transpose_1/strided_slice:output:0'model/conv_transpose_1/stack/1:output:0'model/conv_transpose_1/stack/2:output:0'model/conv_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_1/stack?
,model/conv_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_1/strided_slice_1/stack?
.model/conv_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_1/strided_slice_1/stack_1?
.model/conv_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_1/strided_slice_1/stack_2?
&model/conv_transpose_1/strided_slice_1StridedSlice%model/conv_transpose_1/stack:output:05model/conv_transpose_1/strided_slice_1/stack:output:07model/conv_transpose_1/strided_slice_1/stack_1:output:07model/conv_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_1/strided_slice_1?
6model/conv_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype028
6model/conv_transpose_1/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_1/conv2d_transposeConv2DBackpropInput%model/conv_transpose_1/stack:output:0>model/conv_transpose_1/conv2d_transpose/ReadVariableOp:value:0!model/concatenate/concat:output:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2)
'model/conv_transpose_1/conv2d_transpose?
model/bn_1/ReadVariableOpReadVariableOp"model_bn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
model/bn_1/ReadVariableOp?
model/bn_1/ReadVariableOp_1ReadVariableOp$model_bn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model/bn_1/ReadVariableOp_1?
*model/bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp3model_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/bn_1/FusedBatchNormV3/ReadVariableOp?
,model/bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5model_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/bn_1/FusedBatchNormV3/ReadVariableOp_1?
model/bn_1/FusedBatchNormV3FusedBatchNormV30model/conv_transpose_1/conv2d_transpose:output:0!model/bn_1/ReadVariableOp:value:0#model/bn_1/ReadVariableOp_1:value:02model/bn_1/FusedBatchNormV3/ReadVariableOp:value:04model/bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
model/bn_1/FusedBatchNormV3?
model/relu_1/ReluRelumodel/bn_1/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model/relu_1/Relu?
model/conv_transpose_2/ShapeShapemodel/relu_1/Relu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_2/Shape?
*model/conv_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_2/strided_slice/stack?
,model/conv_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_2/strided_slice/stack_1?
,model/conv_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_2/strided_slice/stack_2?
$model/conv_transpose_2/strided_sliceStridedSlice%model/conv_transpose_2/Shape:output:03model/conv_transpose_2/strided_slice/stack:output:05model/conv_transpose_2/strided_slice/stack_1:output:05model/conv_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_2/strided_slice?
model/conv_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_2/stack/1?
model/conv_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_2/stack/2?
model/conv_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_2/stack/3?
model/conv_transpose_2/stackPack-model/conv_transpose_2/strided_slice:output:0'model/conv_transpose_2/stack/1:output:0'model/conv_transpose_2/stack/2:output:0'model/conv_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_2/stack?
,model/conv_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_2/strided_slice_1/stack?
.model/conv_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_2/strided_slice_1/stack_1?
.model/conv_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_2/strided_slice_1/stack_2?
&model/conv_transpose_2/strided_slice_1StridedSlice%model/conv_transpose_2/stack:output:05model/conv_transpose_2/strided_slice_1/stack:output:07model/conv_transpose_2/strided_slice_1/stack_1:output:07model/conv_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_2/strided_slice_1?
6model/conv_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype028
6model/conv_transpose_2/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_2/conv2d_transposeConv2DBackpropInput%model/conv_transpose_2/stack:output:0>model/conv_transpose_2/conv2d_transpose/ReadVariableOp:value:0model/relu_1/Relu:activations:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2)
'model/conv_transpose_2/conv2d_transpose?
model/bn_2/ReadVariableOpReadVariableOp"model_bn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
model/bn_2/ReadVariableOp?
model/bn_2/ReadVariableOp_1ReadVariableOp$model_bn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model/bn_2/ReadVariableOp_1?
*model/bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp3model_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/bn_2/FusedBatchNormV3/ReadVariableOp?
,model/bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5model_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/bn_2/FusedBatchNormV3/ReadVariableOp_1?
model/bn_2/FusedBatchNormV3FusedBatchNormV30model/conv_transpose_2/conv2d_transpose:output:0!model/bn_2/ReadVariableOp:value:0#model/bn_2/ReadVariableOp_1:value:02model/bn_2/FusedBatchNormV3/ReadVariableOp:value:04model/bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
model/bn_2/FusedBatchNormV3?
model/relu_2/ReluRelumodel/bn_2/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:??????????2
model/relu_2/Relu?
model/conv_transpose_3/ShapeShapemodel/relu_2/Relu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_3/Shape?
*model/conv_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_3/strided_slice/stack?
,model/conv_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_3/strided_slice/stack_1?
,model/conv_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_3/strided_slice/stack_2?
$model/conv_transpose_3/strided_sliceStridedSlice%model/conv_transpose_3/Shape:output:03model/conv_transpose_3/strided_slice/stack:output:05model/conv_transpose_3/strided_slice/stack_1:output:05model/conv_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_3/strided_slice?
model/conv_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B : 2 
model/conv_transpose_3/stack/1?
model/conv_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B : 2 
model/conv_transpose_3/stack/2?
model/conv_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_3/stack/3?
model/conv_transpose_3/stackPack-model/conv_transpose_3/strided_slice:output:0'model/conv_transpose_3/stack/1:output:0'model/conv_transpose_3/stack/2:output:0'model/conv_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_3/stack?
,model/conv_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_3/strided_slice_1/stack?
.model/conv_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_3/strided_slice_1/stack_1?
.model/conv_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_3/strided_slice_1/stack_2?
&model/conv_transpose_3/strided_slice_1StridedSlice%model/conv_transpose_3/stack:output:05model/conv_transpose_3/strided_slice_1/stack:output:07model/conv_transpose_3/strided_slice_1/stack_1:output:07model/conv_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_3/strided_slice_1?
6model/conv_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_3_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype028
6model/conv_transpose_3/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_3/conv2d_transposeConv2DBackpropInput%model/conv_transpose_3/stack:output:0>model/conv_transpose_3/conv2d_transpose/ReadVariableOp:value:0model/relu_2/Relu:activations:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2)
'model/conv_transpose_3/conv2d_transpose?
model/bn_3/ReadVariableOpReadVariableOp"model_bn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
model/bn_3/ReadVariableOp?
model/bn_3/ReadVariableOp_1ReadVariableOp$model_bn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model/bn_3/ReadVariableOp_1?
*model/bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp3model_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/bn_3/FusedBatchNormV3/ReadVariableOp?
,model/bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5model_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/bn_3/FusedBatchNormV3/ReadVariableOp_1?
model/bn_3/FusedBatchNormV3FusedBatchNormV30model/conv_transpose_3/conv2d_transpose:output:0!model/bn_3/ReadVariableOp:value:0#model/bn_3/ReadVariableOp_1:value:02model/bn_3/FusedBatchNormV3/ReadVariableOp:value:04model/bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
model/bn_3/FusedBatchNormV3?
model/relu_3/ReluRelumodel/bn_3/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????  ?2
model/relu_3/Relu?
model/conv_transpose_4/ShapeShapemodel/relu_3/Relu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_4/Shape?
*model/conv_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_4/strided_slice/stack?
,model/conv_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_4/strided_slice/stack_1?
,model/conv_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_4/strided_slice/stack_2?
$model/conv_transpose_4/strided_sliceStridedSlice%model/conv_transpose_4/Shape:output:03model/conv_transpose_4/strided_slice/stack:output:05model/conv_transpose_4/strided_slice/stack_1:output:05model/conv_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_4/strided_slice?
model/conv_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_4/stack/1?
model/conv_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :@2 
model/conv_transpose_4/stack/2?
model/conv_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_4/stack/3?
model/conv_transpose_4/stackPack-model/conv_transpose_4/strided_slice:output:0'model/conv_transpose_4/stack/1:output:0'model/conv_transpose_4/stack/2:output:0'model/conv_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_4/stack?
,model/conv_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_4/strided_slice_1/stack?
.model/conv_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_4/strided_slice_1/stack_1?
.model/conv_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_4/strided_slice_1/stack_2?
&model/conv_transpose_4/strided_slice_1StridedSlice%model/conv_transpose_4/stack:output:05model/conv_transpose_4/strided_slice_1/stack:output:07model/conv_transpose_4/strided_slice_1/stack_1:output:07model/conv_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_4/strided_slice_1?
6model/conv_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_4_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype028
6model/conv_transpose_4/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_4/conv2d_transposeConv2DBackpropInput%model/conv_transpose_4/stack:output:0>model/conv_transpose_4/conv2d_transpose/ReadVariableOp:value:0model/relu_3/Relu:activations:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2)
'model/conv_transpose_4/conv2d_transpose?
model/bn_4/ReadVariableOpReadVariableOp"model_bn_4_readvariableop_resource*
_output_shapes	
:?*
dtype02
model/bn_4/ReadVariableOp?
model/bn_4/ReadVariableOp_1ReadVariableOp$model_bn_4_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model/bn_4/ReadVariableOp_1?
*model/bn_4/FusedBatchNormV3/ReadVariableOpReadVariableOp3model_bn_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02,
*model/bn_4/FusedBatchNormV3/ReadVariableOp?
,model/bn_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp5model_bn_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02.
,model/bn_4/FusedBatchNormV3/ReadVariableOp_1?
model/bn_4/FusedBatchNormV3FusedBatchNormV30model/conv_transpose_4/conv2d_transpose:output:0!model/bn_4/ReadVariableOp:value:0#model/bn_4/ReadVariableOp_1:value:02model/bn_4/FusedBatchNormV3/ReadVariableOp:value:04model/bn_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????@@?:?:?:?:?:*
epsilon%??L?*
is_training( 2
model/bn_4/FusedBatchNormV3?
model/relu_4/ReluRelumodel/bn_4/FusedBatchNormV3:y:0*
T0*0
_output_shapes
:?????????@@?2
model/relu_4/Relu?
model/conv_transpose_5/ShapeShapemodel/relu_4/Relu:activations:0*
T0*
_output_shapes
:2
model/conv_transpose_5/Shape?
*model/conv_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv_transpose_5/strided_slice/stack?
,model/conv_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_5/strided_slice/stack_1?
,model/conv_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv_transpose_5/strided_slice/stack_2?
$model/conv_transpose_5/strided_sliceStridedSlice%model/conv_transpose_5/Shape:output:03model/conv_transpose_5/strided_slice/stack:output:05model/conv_transpose_5/strided_slice/stack_1:output:05model/conv_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv_transpose_5/strided_slice?
model/conv_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_5/stack/1?
model/conv_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv_transpose_5/stack/2?
model/conv_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv_transpose_5/stack/3?
model/conv_transpose_5/stackPack-model/conv_transpose_5/strided_slice:output:0'model/conv_transpose_5/stack/1:output:0'model/conv_transpose_5/stack/2:output:0'model/conv_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv_transpose_5/stack?
,model/conv_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv_transpose_5/strided_slice_1/stack?
.model/conv_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_5/strided_slice_1/stack_1?
.model/conv_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv_transpose_5/strided_slice_1/stack_2?
&model/conv_transpose_5/strided_slice_1StridedSlice%model/conv_transpose_5/stack:output:05model/conv_transpose_5/strided_slice_1/stack:output:07model/conv_transpose_5/strided_slice_1/stack_1:output:07model/conv_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv_transpose_5/strided_slice_1?
6model/conv_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv_transpose_5_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype028
6model/conv_transpose_5/conv2d_transpose/ReadVariableOp?
'model/conv_transpose_5/conv2d_transposeConv2DBackpropInput%model/conv_transpose_5/stack:output:0>model/conv_transpose_5/conv2d_transpose/ReadVariableOp:value:0model/relu_4/Relu:activations:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2)
'model/conv_transpose_5/conv2d_transpose?
model/conv_transpose_5/TanhTanh0model/conv_transpose_5/conv2d_transpose:output:0*
T0*1
_output_shapes
:???????????2
model/conv_transpose_5/Tanh?
IdentityIdentitymodel/conv_transpose_5/Tanh:y:0^NoOp*
T0*1
_output_shapes
:???????????2

Identity?
NoOpNoOp+^model/bn_1/FusedBatchNormV3/ReadVariableOp-^model/bn_1/FusedBatchNormV3/ReadVariableOp_1^model/bn_1/ReadVariableOp^model/bn_1/ReadVariableOp_1+^model/bn_2/FusedBatchNormV3/ReadVariableOp-^model/bn_2/FusedBatchNormV3/ReadVariableOp_1^model/bn_2/ReadVariableOp^model/bn_2/ReadVariableOp_1+^model/bn_3/FusedBatchNormV3/ReadVariableOp-^model/bn_3/FusedBatchNormV3/ReadVariableOp_1^model/bn_3/ReadVariableOp^model/bn_3/ReadVariableOp_1+^model/bn_4/FusedBatchNormV3/ReadVariableOp-^model/bn_4/FusedBatchNormV3/ReadVariableOp_1^model/bn_4/ReadVariableOp^model/bn_4/ReadVariableOp_17^model/conv_transpose_1/conv2d_transpose/ReadVariableOp7^model/conv_transpose_2/conv2d_transpose/ReadVariableOp7^model/conv_transpose_3/conv2d_transpose/ReadVariableOp7^model/conv_transpose_4/conv2d_transpose/ReadVariableOp7^model/conv_transpose_5/conv2d_transpose/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp!^model/embedding/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:?????????d: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*model/bn_1/FusedBatchNormV3/ReadVariableOp*model/bn_1/FusedBatchNormV3/ReadVariableOp2\
,model/bn_1/FusedBatchNormV3/ReadVariableOp_1,model/bn_1/FusedBatchNormV3/ReadVariableOp_126
model/bn_1/ReadVariableOpmodel/bn_1/ReadVariableOp2:
model/bn_1/ReadVariableOp_1model/bn_1/ReadVariableOp_12X
*model/bn_2/FusedBatchNormV3/ReadVariableOp*model/bn_2/FusedBatchNormV3/ReadVariableOp2\
,model/bn_2/FusedBatchNormV3/ReadVariableOp_1,model/bn_2/FusedBatchNormV3/ReadVariableOp_126
model/bn_2/ReadVariableOpmodel/bn_2/ReadVariableOp2:
model/bn_2/ReadVariableOp_1model/bn_2/ReadVariableOp_12X
*model/bn_3/FusedBatchNormV3/ReadVariableOp*model/bn_3/FusedBatchNormV3/ReadVariableOp2\
,model/bn_3/FusedBatchNormV3/ReadVariableOp_1,model/bn_3/FusedBatchNormV3/ReadVariableOp_126
model/bn_3/ReadVariableOpmodel/bn_3/ReadVariableOp2:
model/bn_3/ReadVariableOp_1model/bn_3/ReadVariableOp_12X
*model/bn_4/FusedBatchNormV3/ReadVariableOp*model/bn_4/FusedBatchNormV3/ReadVariableOp2\
,model/bn_4/FusedBatchNormV3/ReadVariableOp_1,model/bn_4/FusedBatchNormV3/ReadVariableOp_126
model/bn_4/ReadVariableOpmodel/bn_4/ReadVariableOp2:
model/bn_4/ReadVariableOp_1model/bn_4/ReadVariableOp_12p
6model/conv_transpose_1/conv2d_transpose/ReadVariableOp6model/conv_transpose_1/conv2d_transpose/ReadVariableOp2p
6model/conv_transpose_2/conv2d_transpose/ReadVariableOp6model/conv_transpose_2/conv2d_transpose/ReadVariableOp2p
6model/conv_transpose_3/conv2d_transpose/ReadVariableOp6model/conv_transpose_3/conv2d_transpose/ReadVariableOp2p
6model/conv_transpose_4/conv2d_transpose/ReadVariableOp6model/conv_transpose_4/conv2d_transpose/ReadVariableOp2p
6model/conv_transpose_5/conv2d_transpose/ReadVariableOp6model/conv_transpose_5/conv2d_transpose/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2D
 model/embedding/embedding_lookup model/embedding/embedding_lookup:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1:PL
'
_output_shapes
:?????????d
!
_user_specified_name	input_2
?
E
)__inference_reshape_layer_call_fn_2629491

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:?????????2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"?-
saver_filename:0
Identity:0Identity_278"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????
;
input_20
serving_default_input_2:0?????????dN
conv_transpose_5:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??
?
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer-7
	layer-8

layer_with_weights-3

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer_with_weights-10
layer-19
layer-20
layer_with_weights-11
layer-21

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
D
#_self_saveable_object_factories"
_tf_keras_input_layer
?

kernel
 bias
#!_self_saveable_object_factories
"trainable_variables
#	variables
$regularization_losses
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
&
embeddings
#'_self_saveable_object_factories
(trainable_variables
)	variables
*regularization_losses
+	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#,_self_saveable_object_factories
-trainable_variables
.	variables
/regularization_losses
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

1kernel
2bias
#3_self_saveable_object_factories
4trainable_variables
5	variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#8_self_saveable_object_factories
9trainable_variables
:	variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#=_self_saveable_object_factories
>trainable_variables
?	variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#B_self_saveable_object_factories
Ctrainable_variables
D	variables
Eregularization_losses
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Gkernel
#H_self_saveable_object_factories
Itrainable_variables
J	variables
Kregularization_losses
L	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Maxis
	Ngamma
Obeta
Pmoving_mean
Qmoving_variance
#R_self_saveable_object_factories
Strainable_variables
T	variables
Uregularization_losses
V	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#W_self_saveable_object_factories
Xtrainable_variables
Y	variables
Zregularization_losses
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

\kernel
#]_self_saveable_object_factories
^trainable_variables
_	variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
baxis
	cgamma
dbeta
emoving_mean
fmoving_variance
#g_self_saveable_object_factories
htrainable_variables
i	variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#l_self_saveable_object_factories
mtrainable_variables
n	variables
oregularization_losses
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

qkernel
#r_self_saveable_object_factories
strainable_variables
t	variables
uregularization_losses
v	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
#|_self_saveable_object_factories
}trainable_variables
~	variables
regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
	?axis

?gamma
	?beta
?moving_mean
?moving_variance
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
?kernel
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
-
?serving_default"
signature_map
 "
trackable_dict_wrapper
?
0
 1
&2
13
24
G5
N6
O7
\8
c9
d10
q11
x12
y13
?14
?15
?16
?17"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
 1
&2
13
24
G5
N6
O7
P8
Q9
\10
c11
d12
e13
f14
q15
x16
y17
z18
{19
?20
?21
?22
?23
?24
?25"
trackable_list_wrapper
?
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
!:	d?@2dense_1/kernel
:?@2dense_1/bias
 "
trackable_dict_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"trainable_variables
?layer_metrics
#	variables
 ?layer_regularization_losses
$regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$2embedding/embeddings
 "
trackable_dict_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
(trainable_variables
?layer_metrics
)	variables
 ?layer_regularization_losses
*regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
-trainable_variables
?layer_metrics
.	variables
 ?layer_regularization_losses
/regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:2dense/kernel
:2
dense/bias
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4trainable_variables
?layer_metrics
5	variables
 ?layer_regularization_losses
6regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
9trainable_variables
?layer_metrics
:	variables
 ?layer_regularization_losses
;regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
>trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
@regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ctrainable_variables
?layer_metrics
D	variables
 ?layer_regularization_losses
Eregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_1/kernel
 "
trackable_dict_wrapper
'
G0"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Itrainable_variables
?layer_metrics
J	variables
 ?layer_regularization_losses
Kregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2
bn_1/gamma
:?2	bn_1/beta
!:? (2bn_1/moving_mean
%:#? (2bn_1/moving_variance
 "
trackable_dict_wrapper
.
N0
O1"
trackable_list_wrapper
<
N0
O1
P2
Q3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Strainable_variables
?layer_metrics
T	variables
 ?layer_regularization_losses
Uregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xtrainable_variables
?layer_metrics
Y	variables
 ?layer_regularization_losses
Zregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_2/kernel
 "
trackable_dict_wrapper
'
\0"
trackable_list_wrapper
'
\0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
^trainable_variables
?layer_metrics
_	variables
 ?layer_regularization_losses
`regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2
bn_2/gamma
:?2	bn_2/beta
!:? (2bn_2/moving_mean
%:#? (2bn_2/moving_variance
 "
trackable_dict_wrapper
.
c0
d1"
trackable_list_wrapper
<
c0
d1
e2
f3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
htrainable_variables
?layer_metrics
i	variables
 ?layer_regularization_losses
jregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
mtrainable_variables
?layer_metrics
n	variables
 ?layer_regularization_losses
oregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_3/kernel
 "
trackable_dict_wrapper
'
q0"
trackable_list_wrapper
'
q0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
strainable_variables
?layer_metrics
t	variables
 ?layer_regularization_losses
uregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2
bn_3/gamma
:?2	bn_3/beta
!:? (2bn_3/moving_mean
%:#? (2bn_3/moving_variance
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
}trainable_variables
?layer_metrics
~	variables
 ?layer_regularization_losses
regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_4/kernel
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2
bn_4/gamma
:?2	bn_4/beta
!:? (2bn_4/moving_mean
%:#? (2bn_4/moving_variance
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
@
?0
?1
?2
?3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0?2conv_transpose_5/kernel
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?trainable_variables
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
Z
P0
Q1
e2
f3
z4
{5
?6
?7"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
'__inference_model_layer_call_fn_2626269
'__inference_model_layer_call_fn_2628718
'__inference_model_layer_call_fn_2628925
'__inference_model_layer_call_fn_2628037?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
"__inference__wrapped_model_2624863input_1input_2"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_model_layer_call_and_return_conditional_losses_2629132
B__inference_model_layer_call_and_return_conditional_losses_2629339
B__inference_model_layer_call_and_return_conditional_losses_2628244
B__inference_model_layer_call_and_return_conditional_losses_2628451?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
)__inference_dense_1_layer_call_fn_2629349?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_dense_1_layer_call_and_return_conditional_losses_2629359?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_embedding_layer_call_fn_2629369?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_embedding_layer_call_and_return_conditional_losses_2629379?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_re_lu_layer_call_fn_2629384?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_re_lu_layer_call_and_return_conditional_losses_2629389?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_layer_call_fn_2629419?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_layer_call_and_return_conditional_losses_2629449?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_reshape_1_layer_call_fn_2629463?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_reshape_1_layer_call_and_return_conditional_losses_2629477?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
)__inference_reshape_layer_call_fn_2629491?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_reshape_layer_call_and_return_conditional_losses_2629505?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
-__inference_concatenate_layer_call_fn_2629512?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
H__inference_concatenate_layer_call_and_return_conditional_losses_2629519?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv_transpose_1_layer_call_fn_2629549
2__inference_conv_transpose_1_layer_call_fn_2629569?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629599
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629619?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_bn_1_layer_call_fn_2629637
&__inference_bn_1_layer_call_fn_2629655
&__inference_bn_1_layer_call_fn_2629673
&__inference_bn_1_layer_call_fn_2629691?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629709
A__inference_bn_1_layer_call_and_return_conditional_losses_2629727
A__inference_bn_1_layer_call_and_return_conditional_losses_2629745
A__inference_bn_1_layer_call_and_return_conditional_losses_2629763?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_relu_1_layer_call_fn_2629768?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_relu_1_layer_call_and_return_conditional_losses_2629773?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv_transpose_2_layer_call_fn_2629803
2__inference_conv_transpose_2_layer_call_fn_2629823?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629853
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_bn_2_layer_call_fn_2629891
&__inference_bn_2_layer_call_fn_2629909
&__inference_bn_2_layer_call_fn_2629927
&__inference_bn_2_layer_call_fn_2629945?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629963
A__inference_bn_2_layer_call_and_return_conditional_losses_2629981
A__inference_bn_2_layer_call_and_return_conditional_losses_2629999
A__inference_bn_2_layer_call_and_return_conditional_losses_2630017?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_relu_2_layer_call_fn_2630022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_relu_2_layer_call_and_return_conditional_losses_2630027?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv_transpose_3_layer_call_fn_2630057
2__inference_conv_transpose_3_layer_call_fn_2630077?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630107
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630127?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_bn_3_layer_call_fn_2630145
&__inference_bn_3_layer_call_fn_2630163
&__inference_bn_3_layer_call_fn_2630181
&__inference_bn_3_layer_call_fn_2630199?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630217
A__inference_bn_3_layer_call_and_return_conditional_losses_2630235
A__inference_bn_3_layer_call_and_return_conditional_losses_2630253
A__inference_bn_3_layer_call_and_return_conditional_losses_2630271?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_relu_3_layer_call_fn_2630276?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_relu_3_layer_call_and_return_conditional_losses_2630281?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv_transpose_4_layer_call_fn_2630311
2__inference_conv_transpose_4_layer_call_fn_2630331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630361
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630381?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_bn_4_layer_call_fn_2630399
&__inference_bn_4_layer_call_fn_2630417
&__inference_bn_4_layer_call_fn_2630435
&__inference_bn_4_layer_call_fn_2630453?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630471
A__inference_bn_4_layer_call_and_return_conditional_losses_2630489
A__inference_bn_4_layer_call_and_return_conditional_losses_2630507
A__inference_bn_4_layer_call_and_return_conditional_losses_2630525?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
(__inference_relu_4_layer_call_fn_2630530?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_relu_4_layer_call_and_return_conditional_losses_2630535?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
2__inference_conv_transpose_5_layer_call_fn_2630566
2__inference_conv_transpose_5_layer_call_fn_2630587?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630618
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630639?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
%__inference_signature_wrapper_2628511input_1input_2"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
"__inference__wrapped_model_2624863? & 12GNOPQ\cdefqxyz{??????X?U
N?K
I?F
!?
input_1?????????
!?
input_2?????????d
? "M?J
H
conv_transpose_54?1
conv_transpose_5????????????
A__inference_bn_1_layer_call_and_return_conditional_losses_2629709?NOPQN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629727?NOPQN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629745tNOPQ<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2629763tNOPQ<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_bn_1_layer_call_fn_2629637?NOPQN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_1_layer_call_fn_2629655?NOPQN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_1_layer_call_fn_2629673gNOPQ<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_bn_1_layer_call_fn_2629691gNOPQ<?9
2?/
)?&
inputs??????????
p
? "!????????????
A__inference_bn_2_layer_call_and_return_conditional_losses_2629963?cdefN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629981?cdefN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2629999tcdef<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2630017tcdef<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_bn_2_layer_call_fn_2629891?cdefN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_2_layer_call_fn_2629909?cdefN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_2_layer_call_fn_2629927gcdef<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_bn_2_layer_call_fn_2629945gcdef<?9
2?/
)?&
inputs??????????
p
? "!????????????
A__inference_bn_3_layer_call_and_return_conditional_losses_2630217?xyz{N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630235?xyz{N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630253txyz{<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2630271txyz{<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
&__inference_bn_3_layer_call_fn_2630145?xyz{N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_3_layer_call_fn_2630163?xyz{N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_3_layer_call_fn_2630181gxyz{<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
&__inference_bn_3_layer_call_fn_2630199gxyz{<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
A__inference_bn_4_layer_call_and_return_conditional_losses_2630471?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630489?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630507x????<?9
2?/
)?&
inputs?????????@@?
p 
? ".?+
$?!
0?????????@@?
? ?
A__inference_bn_4_layer_call_and_return_conditional_losses_2630525x????<?9
2?/
)?&
inputs?????????@@?
p
? ".?+
$?!
0?????????@@?
? ?
&__inference_bn_4_layer_call_fn_2630399?????N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_4_layer_call_fn_2630417?????N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_4_layer_call_fn_2630435k????<?9
2?/
)?&
inputs?????????@@?
p 
? "!??????????@@??
&__inference_bn_4_layer_call_fn_2630453k????<?9
2?/
)?&
inputs?????????@@?
p
? "!??????????@@??
H__inference_concatenate_layer_call_and_return_conditional_losses_2629519?k?h
a?^
\?Y
+?(
inputs/0??????????
*?'
inputs/1?????????
? ".?+
$?!
0??????????
? ?
-__inference_concatenate_layer_call_fn_2629512?k?h
a?^
\?Y
+?(
inputs/0??????????
*?'
inputs/1?????????
? "!????????????
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629599?GJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2629619mG8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
2__inference_conv_transpose_1_layer_call_fn_2629549?GJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
2__inference_conv_transpose_1_layer_call_fn_2629569`G8?5
.?+
)?&
inputs??????????
? "!????????????
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629853?\J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2629873m\8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
2__inference_conv_transpose_2_layer_call_fn_2629803?\J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
2__inference_conv_transpose_2_layer_call_fn_2629823`\8?5
.?+
)?&
inputs??????????
? "!????????????
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630107?qJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2630127mq8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????  ?
? ?
2__inference_conv_transpose_3_layer_call_fn_2630057?qJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
2__inference_conv_transpose_3_layer_call_fn_2630077`q8?5
.?+
)?&
inputs??????????
? "!??????????  ??
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630361??J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2630381n?8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????@@?
? ?
2__inference_conv_transpose_4_layer_call_fn_2630311??J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
2__inference_conv_transpose_4_layer_call_fn_2630331a?8?5
.?+
)?&
inputs?????????  ?
? "!??????????@@??
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630618??J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
M__inference_conv_transpose_5_layer_call_and_return_conditional_losses_2630639o?8?5
.?+
)?&
inputs?????????@@?
? "/?,
%?"
0???????????
? ?
2__inference_conv_transpose_5_layer_call_fn_2630566??J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
2__inference_conv_transpose_5_layer_call_fn_2630587b?8?5
.?+
)?&
inputs?????????@@?
? ""?????????????
D__inference_dense_1_layer_call_and_return_conditional_losses_2629359] /?,
%?"
 ?
inputs?????????d
? "&?#
?
0??????????@
? }
)__inference_dense_1_layer_call_fn_2629349P /?,
%?"
 ?
inputs?????????d
? "???????????@?
B__inference_dense_layer_call_and_return_conditional_losses_2629449d123?0
)?&
$?!
inputs?????????
? ")?&
?
0?????????
? ?
'__inference_dense_layer_call_fn_2629419W123?0
)?&
$?!
inputs?????????
? "???????????
F__inference_embedding_layer_call_and_return_conditional_losses_2629379_&/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
+__inference_embedding_layer_call_fn_2629369R&/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_model_layer_call_and_return_conditional_losses_2628244? & 12GNOPQ\cdefqxyz{??????`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????d
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2628451? & 12GNOPQ\cdefqxyz{??????`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????d
p

 
? "/?,
%?"
0???????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2629132? & 12GNOPQ\cdefqxyz{??????b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????d
p 

 
? "/?,
%?"
0???????????
? ?
B__inference_model_layer_call_and_return_conditional_losses_2629339? & 12GNOPQ\cdefqxyz{??????b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????d
p

 
? "/?,
%?"
0???????????
? ?
'__inference_model_layer_call_fn_2626269? & 12GNOPQ\cdefqxyz{??????`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????d
p 

 
? ""?????????????
'__inference_model_layer_call_fn_2628037? & 12GNOPQ\cdefqxyz{??????`?]
V?S
I?F
!?
input_1?????????
!?
input_2?????????d
p

 
? ""?????????????
'__inference_model_layer_call_fn_2628718? & 12GNOPQ\cdefqxyz{??????b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????d
p 

 
? ""?????????????
'__inference_model_layer_call_fn_2628925? & 12GNOPQ\cdefqxyz{??????b?_
X?U
K?H
"?
inputs/0?????????
"?
inputs/1?????????d
p

 
? ""?????????????
B__inference_re_lu_layer_call_and_return_conditional_losses_2629389Z0?-
&?#
!?
inputs??????????@
? "&?#
?
0??????????@
? x
'__inference_re_lu_layer_call_fn_2629384M0?-
&?#
!?
inputs??????????@
? "???????????@?
C__inference_relu_1_layer_call_and_return_conditional_losses_2629773j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
(__inference_relu_1_layer_call_fn_2629768]8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_relu_2_layer_call_and_return_conditional_losses_2630027j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
(__inference_relu_2_layer_call_fn_2630022]8?5
.?+
)?&
inputs??????????
? "!????????????
C__inference_relu_3_layer_call_and_return_conditional_losses_2630281j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
(__inference_relu_3_layer_call_fn_2630276]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
C__inference_relu_4_layer_call_and_return_conditional_losses_2630535j8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????@@?
? ?
(__inference_relu_4_layer_call_fn_2630530]8?5
.?+
)?&
inputs?????????@@?
? "!??????????@@??
F__inference_reshape_1_layer_call_and_return_conditional_losses_2629477b0?-
&?#
!?
inputs??????????@
? ".?+
$?!
0??????????
? ?
+__inference_reshape_1_layer_call_fn_2629463U0?-
&?#
!?
inputs??????????@
? "!????????????
D__inference_reshape_layer_call_and_return_conditional_losses_2629505d3?0
)?&
$?!
inputs?????????
? "-?*
#? 
0?????????
? ?
)__inference_reshape_layer_call_fn_2629491W3?0
)?&
$?!
inputs?????????
? " ???????????
%__inference_signature_wrapper_2628511? & 12GNOPQ\cdefqxyz{??????i?f
? 
_?\
,
input_1!?
input_1?????????
,
input_2!?
input_2?????????d"M?J
H
conv_transpose_54?1
conv_transpose_5???????????