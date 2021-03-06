??
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
Conv2D

input"T
filter"T
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
\
	LeakyRelu
features"T
activations"T"
alphafloat%??L>"
Ttype0:
2
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
0
Sigmoid
x"T
y"T"
Ttype:

2
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
 ?"serve*2.6.02v2.6.0-rc2-32-g919f693420e8??
?
embedding_1/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*'
shared_nameembedding_1/embeddings
?
*embedding_1/embeddings/Read/ReadVariableOpReadVariableOpembedding_1/embeddings*
_output_shapes

:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
??*
dtype0
r
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_namedense_2/bias
k
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes

:??*
dtype0
?
conv_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv_transpose_1/kernel
?
+conv_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_1/kernel*'
_output_shapes
:?*
dtype0
?
conv_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_2/kernel
?
+conv_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_2/kernel*(
_output_shapes
:??*
dtype0
m

bn_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_1/gamma
f
bn_1/gamma/Read/ReadVariableOpReadVariableOp
bn_1/gamma*
_output_shapes	
:?*
dtype0
k
	bn_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_1/beta
d
bn_1/beta/Read/ReadVariableOpReadVariableOp	bn_1/beta*
_output_shapes	
:?*
dtype0
y
bn_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_1/moving_mean
r
$bn_1/moving_mean/Read/ReadVariableOpReadVariableOpbn_1/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_1/moving_variance
z
(bn_1/moving_variance/Read/ReadVariableOpReadVariableOpbn_1/moving_variance*
_output_shapes	
:?*
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
conv_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv_transpose_4/kernel
?
+conv_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv_transpose_4/kernel*(
_output_shapes
:??*
dtype0
m

bn_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name
bn_3/gamma
f
bn_3/gamma/Read/ReadVariableOpReadVariableOp
bn_3/gamma*
_output_shapes	
:?*
dtype0
k
	bn_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_name	bn_3/beta
d
bn_3/beta/Read/ReadVariableOpReadVariableOp	bn_3/beta*
_output_shapes	
:?*
dtype0
y
bn_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*!
shared_namebn_3/moving_mean
r
$bn_3/moving_mean/Read/ReadVariableOpReadVariableOpbn_3/moving_mean*
_output_shapes	
:?*
dtype0
?
bn_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_namebn_3/moving_variance
z
(bn_3/moving_variance/Read/ReadVariableOpReadVariableOpbn_3/moving_variance*
_output_shapes	
:?*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
??*
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?M
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?L
value?LB?L B?L
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
%
#_self_saveable_object_factories
?

embeddings
#_self_saveable_object_factories
trainable_variables
	variables
 regularization_losses
!	keras_api
?

"kernel
#bias
#$_self_saveable_object_factories
%trainable_variables
&	variables
'regularization_losses
(	keras_api
%
#)_self_saveable_object_factories
w
#*_self_saveable_object_factories
+trainable_variables
,	variables
-regularization_losses
.	keras_api
w
#/_self_saveable_object_factories
0trainable_variables
1	variables
2regularization_losses
3	keras_api
?

4kernel
#5_self_saveable_object_factories
6trainable_variables
7	variables
8regularization_losses
9	keras_api
w
#:_self_saveable_object_factories
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?

?kernel
#@_self_saveable_object_factories
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
#J_self_saveable_object_factories
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
w
#O_self_saveable_object_factories
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
?

Tkernel
#U_self_saveable_object_factories
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
#__self_saveable_object_factories
`trainable_variables
a	variables
bregularization_losses
c	keras_api
w
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?

ikernel
#j_self_saveable_object_factories
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
?
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
#t_self_saveable_object_factories
utrainable_variables
v	variables
wregularization_losses
x	keras_api
w
#y_self_saveable_object_factories
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
z
#~_self_saveable_object_factories
trainable_variables
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
	?bias
$?_self_saveable_object_factories
?trainable_variables
?	variables
?regularization_losses
?	keras_api
 
 
p
0
"1
#2
43
?4
F5
G6
T7
[8
\9
i10
p11
q12
?13
?14
 
?
0
"1
#2
43
?4
F5
G6
H7
I8
T9
[10
\11
]12
^13
i14
p15
q16
r17
s18
?19
?20
?
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?metrics
	variables
 
fd
VARIABLE_VALUEembedding_1/embeddings:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
 
?
trainable_variables
?layer_metrics
	variables
 ?layer_regularization_losses
 regularization_losses
?layers
?metrics
?non_trainable_variables
ZX
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
 
?
%trainable_variables
?layer_metrics
&	variables
 ?layer_regularization_losses
'regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
 
?
+trainable_variables
?layer_metrics
,	variables
 ?layer_regularization_losses
-regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
0trainable_variables
?layer_metrics
1	variables
 ?layer_regularization_losses
2regularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

40

40
 
?
6trainable_variables
?layer_metrics
7	variables
 ?layer_regularization_losses
8regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
;trainable_variables
?layer_metrics
<	variables
 ?layer_regularization_losses
=regularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

?0

?0
 
?
Atrainable_variables
?layer_metrics
B	variables
 ?layer_regularization_losses
Cregularization_losses
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
F0
G1

F0
G1
H2
I3
 
?
Ktrainable_variables
?layer_metrics
L	variables
 ?layer_regularization_losses
Mregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
Ptrainable_variables
?layer_metrics
Q	variables
 ?layer_regularization_losses
Rregularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_3/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

T0

T0
 
?
Vtrainable_variables
?layer_metrics
W	variables
 ?layer_regularization_losses
Xregularization_losses
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
[0
\1

[0
\1
]2
^3
 
?
`trainable_variables
?layer_metrics
a	variables
 ?layer_regularization_losses
bregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
etrainable_variables
?layer_metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
?metrics
?non_trainable_variables
ca
VARIABLE_VALUEconv_transpose_4/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
 

i0

i0
 
?
ktrainable_variables
?layer_metrics
l	variables
 ?layer_regularization_losses
mregularization_losses
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
p0
q1

p0
q1
r2
s3
 
?
utrainable_variables
?layer_metrics
v	variables
 ?layer_regularization_losses
wregularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
ztrainable_variables
?layer_metrics
{	variables
 ?layer_regularization_losses
|regularization_losses
?layers
?metrics
?non_trainable_variables
 
 
 
 
?
trainable_variables
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
?layer_metrics
?	variables
 ?layer_regularization_losses
?regularization_losses
?layers
?metrics
?non_trainable_variables
ZX
VARIABLE_VALUEdense_3/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_3/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?0
?1
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
*
H0
I1
]2
^3
r4
s5
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
H0
I1
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
]0
^1
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
r0
s1
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
z
serving_default_input_3Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
serving_default_input_4Placeholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3serving_default_input_4embedding_1/embeddingsdense_2/kerneldense_2/biasconv_transpose_1/kernelconv_transpose_2/kernel
bn_1/gamma	bn_1/betabn_1/moving_meanbn_1/moving_varianceconv_transpose_3/kernel
bn_2/gamma	bn_2/betabn_2/moving_meanbn_2/moving_varianceconv_transpose_4/kernel
bn_3/gamma	bn_3/betabn_3/moving_meanbn_3/moving_variancedense_3/kerneldense_3/bias*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *.
f)R'
%__inference_signature_wrapper_2632987
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
:*
dtype0*?	
value?	B?	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
?
SaveV2SaveV2ShardedFilenameSaveV2/tensor_namesSaveV2/shape_and_slices*embedding_1/embeddings/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp+conv_transpose_1/kernel/Read/ReadVariableOp+conv_transpose_2/kernel/Read/ReadVariableOpbn_1/gamma/Read/ReadVariableOpbn_1/beta/Read/ReadVariableOp$bn_1/moving_mean/Read/ReadVariableOp(bn_1/moving_variance/Read/ReadVariableOp+conv_transpose_3/kernel/Read/ReadVariableOpbn_2/gamma/Read/ReadVariableOpbn_2/beta/Read/ReadVariableOp$bn_2/moving_mean/Read/ReadVariableOp(bn_2/moving_variance/Read/ReadVariableOp+conv_transpose_4/kernel/Read/ReadVariableOpbn_3/gamma/Read/ReadVariableOpbn_3/beta/Read/ReadVariableOp$bn_3/moving_mean/Read/ReadVariableOp(bn_3/moving_variance/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOpConst"/device:CPU:0*$
dtypes
2
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
:*
dtype0*?	
value?	B?	B:layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-6/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-6/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-6/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-8/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-8/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-8/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value6B4B B B B B B B B B B B B B B B B B B B B B B 
?
	RestoreV2	RestoreV2saver_filenameRestoreV2/tensor_namesRestoreV2/shape_and_slices"/device:CPU:0*l
_output_shapesZ
X::::::::::::::::::::::*$
dtypes
2
S

Identity_1Identity	RestoreV2"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOpAssignVariableOpembedding_1/embeddings
Identity_1"/device:CPU:0*
dtype0
U

Identity_2IdentityRestoreV2:1"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_1AssignVariableOpdense_2/kernel
Identity_2"/device:CPU:0*
dtype0
U

Identity_3IdentityRestoreV2:2"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_2AssignVariableOpdense_2/bias
Identity_3"/device:CPU:0*
dtype0
U

Identity_4IdentityRestoreV2:3"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_3AssignVariableOpconv_transpose_1/kernel
Identity_4"/device:CPU:0*
dtype0
U

Identity_5IdentityRestoreV2:4"/device:CPU:0*
T0*
_output_shapes
:
g
AssignVariableOp_4AssignVariableOpconv_transpose_2/kernel
Identity_5"/device:CPU:0*
dtype0
U

Identity_6IdentityRestoreV2:5"/device:CPU:0*
T0*
_output_shapes
:
Z
AssignVariableOp_5AssignVariableOp
bn_1/gamma
Identity_6"/device:CPU:0*
dtype0
U

Identity_7IdentityRestoreV2:6"/device:CPU:0*
T0*
_output_shapes
:
Y
AssignVariableOp_6AssignVariableOp	bn_1/beta
Identity_7"/device:CPU:0*
dtype0
U

Identity_8IdentityRestoreV2:7"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_7AssignVariableOpbn_1/moving_mean
Identity_8"/device:CPU:0*
dtype0
U

Identity_9IdentityRestoreV2:8"/device:CPU:0*
T0*
_output_shapes
:
d
AssignVariableOp_8AssignVariableOpbn_1/moving_variance
Identity_9"/device:CPU:0*
dtype0
V
Identity_10IdentityRestoreV2:9"/device:CPU:0*
T0*
_output_shapes
:
h
AssignVariableOp_9AssignVariableOpconv_transpose_3/kernelIdentity_10"/device:CPU:0*
dtype0
W
Identity_11IdentityRestoreV2:10"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_10AssignVariableOp
bn_2/gammaIdentity_11"/device:CPU:0*
dtype0
W
Identity_12IdentityRestoreV2:11"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_11AssignVariableOp	bn_2/betaIdentity_12"/device:CPU:0*
dtype0
W
Identity_13IdentityRestoreV2:12"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_12AssignVariableOpbn_2/moving_meanIdentity_13"/device:CPU:0*
dtype0
W
Identity_14IdentityRestoreV2:13"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_13AssignVariableOpbn_2/moving_varianceIdentity_14"/device:CPU:0*
dtype0
W
Identity_15IdentityRestoreV2:14"/device:CPU:0*
T0*
_output_shapes
:
i
AssignVariableOp_14AssignVariableOpconv_transpose_4/kernelIdentity_15"/device:CPU:0*
dtype0
W
Identity_16IdentityRestoreV2:15"/device:CPU:0*
T0*
_output_shapes
:
\
AssignVariableOp_15AssignVariableOp
bn_3/gammaIdentity_16"/device:CPU:0*
dtype0
W
Identity_17IdentityRestoreV2:16"/device:CPU:0*
T0*
_output_shapes
:
[
AssignVariableOp_16AssignVariableOp	bn_3/betaIdentity_17"/device:CPU:0*
dtype0
W
Identity_18IdentityRestoreV2:17"/device:CPU:0*
T0*
_output_shapes
:
b
AssignVariableOp_17AssignVariableOpbn_3/moving_meanIdentity_18"/device:CPU:0*
dtype0
W
Identity_19IdentityRestoreV2:18"/device:CPU:0*
T0*
_output_shapes
:
f
AssignVariableOp_18AssignVariableOpbn_3/moving_varianceIdentity_19"/device:CPU:0*
dtype0
W
Identity_20IdentityRestoreV2:19"/device:CPU:0*
T0*
_output_shapes
:
`
AssignVariableOp_19AssignVariableOpdense_3/kernelIdentity_20"/device:CPU:0*
dtype0
W
Identity_21IdentityRestoreV2:20"/device:CPU:0*
T0*
_output_shapes
:
^
AssignVariableOp_20AssignVariableOpdense_3/biasIdentity_21"/device:CPU:0*
dtype0

NoOp_1NoOp"/device:CPU:0
?
Identity_22Identitysaver_filename^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp_1"/device:CPU:0*
T0*
_output_shapes
: ??
??
?
D__inference_model_1_layer_call_and_return_conditional_losses_2632813
input_4
input_36
$embedding_1_embedding_lookup_2632701:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupv
embedding_1/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2632701embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632701*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632701*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_4reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape~
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:???????????2
dropout/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2L
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?

?
-__inference_embedding_1_layer_call_fn_2633479

inputs*
embedding_lookup_2633473:
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_2633473Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/2633473*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/2633473*+
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
?
?
2__inference_conv_transpose_3_layer_call_fn_2633790

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2634001

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
?
[
/__inference_concatenate_1_layer_call_fn_2633584
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
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?
?
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2633605

inputs9
conv2d_readvariableop_resource:?
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
ݏ
?
)__inference_model_1_layer_call_fn_2633104
inputs_0
inputs_16
$embedding_1_embedding_lookup_2632992:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2632992embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632992*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632992*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2inputs_0reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape~
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:???????????2
dropout/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2L
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?

?
)__inference_dense_3_layer_call_fn_2634176

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2634037

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
b
D__inference_dropout_layer_call_and_return_conditional_losses_2634153

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_bn_2_layer_call_fn_2633833

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
??
?
"__inference__wrapped_model_2631001
input_4
input_3>
,model_1_embedding_1_embedding_lookup_2630889:E
1model_1_dense_2_tensordot_readvariableop_resource:
???
/model_1_dense_2_biasadd_readvariableop_resource:
??R
7model_1_conv_transpose_1_conv2d_readvariableop_resource:?S
7model_1_conv_transpose_2_conv2d_readvariableop_resource:??3
$model_1_bn_1_readvariableop_resource:	?5
&model_1_bn_1_readvariableop_1_resource:	?D
5model_1_bn_1_fusedbatchnormv3_readvariableop_resource:	?F
7model_1_bn_1_fusedbatchnormv3_readvariableop_1_resource:	?S
7model_1_conv_transpose_3_conv2d_readvariableop_resource:??3
$model_1_bn_2_readvariableop_resource:	?5
&model_1_bn_2_readvariableop_1_resource:	?D
5model_1_bn_2_fusedbatchnormv3_readvariableop_resource:	?F
7model_1_bn_2_fusedbatchnormv3_readvariableop_1_resource:	?S
7model_1_conv_transpose_4_conv2d_readvariableop_resource:??3
$model_1_bn_3_readvariableop_resource:	?5
&model_1_bn_3_readvariableop_1_resource:	?D
5model_1_bn_3_fusedbatchnormv3_readvariableop_resource:	?F
7model_1_bn_3_fusedbatchnormv3_readvariableop_1_resource:	?B
.model_1_dense_3_matmul_readvariableop_resource:
??=
/model_1_dense_3_biasadd_readvariableop_resource:
identity??,model_1/bn_1/FusedBatchNormV3/ReadVariableOp?.model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1?model_1/bn_1/ReadVariableOp?model_1/bn_1/ReadVariableOp_1?,model_1/bn_2/FusedBatchNormV3/ReadVariableOp?.model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1?model_1/bn_2/ReadVariableOp?model_1/bn_2/ReadVariableOp_1?,model_1/bn_3/FusedBatchNormV3/ReadVariableOp?.model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1?model_1/bn_3/ReadVariableOp?model_1/bn_3/ReadVariableOp_1?.model_1/conv_transpose_1/Conv2D/ReadVariableOp?.model_1/conv_transpose_2/Conv2D/ReadVariableOp?.model_1/conv_transpose_3/Conv2D/ReadVariableOp?.model_1/conv_transpose_4/Conv2D/ReadVariableOp?&model_1/dense_2/BiasAdd/ReadVariableOp?(model_1/dense_2/Tensordot/ReadVariableOp?&model_1/dense_3/BiasAdd/ReadVariableOp?%model_1/dense_3/MatMul/ReadVariableOp?$model_1/embedding_1/embedding_lookup?
model_1/embedding_1/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????2
model_1/embedding_1/Cast?
$model_1/embedding_1/embedding_lookupResourceGather,model_1_embedding_1_embedding_lookup_2630889model_1/embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*?
_class5
31loc:@model_1/embedding_1/embedding_lookup/2630889*+
_output_shapes
:?????????*
dtype02&
$model_1/embedding_1/embedding_lookup?
-model_1/embedding_1/embedding_lookup/IdentityIdentity-model_1/embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*?
_class5
31loc:@model_1/embedding_1/embedding_lookup/2630889*+
_output_shapes
:?????????2/
-model_1/embedding_1/embedding_lookup/Identity?
/model_1/embedding_1/embedding_lookup/Identity_1Identity6model_1/embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????21
/model_1/embedding_1/embedding_lookup/Identity_1?
(model_1/dense_2/Tensordot/ReadVariableOpReadVariableOp1model_1_dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02*
(model_1/dense_2/Tensordot/ReadVariableOp?
model_1/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2 
model_1/dense_2/Tensordot/axes?
model_1/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2 
model_1/dense_2/Tensordot/free?
model_1/dense_2/Tensordot/ShapeShape8model_1/embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2!
model_1/dense_2/Tensordot/Shape?
'model_1/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_2/Tensordot/GatherV2/axis?
"model_1/dense_2/Tensordot/GatherV2GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/free:output:00model_1/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2$
"model_1/dense_2/Tensordot/GatherV2?
)model_1/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2+
)model_1/dense_2/Tensordot/GatherV2_1/axis?
$model_1/dense_2/Tensordot/GatherV2_1GatherV2(model_1/dense_2/Tensordot/Shape:output:0'model_1/dense_2/Tensordot/axes:output:02model_1/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2&
$model_1/dense_2/Tensordot/GatherV2_1?
model_1/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2!
model_1/dense_2/Tensordot/Const?
model_1/dense_2/Tensordot/ProdProd+model_1/dense_2/Tensordot/GatherV2:output:0(model_1/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2 
model_1/dense_2/Tensordot/Prod?
!model_1/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2#
!model_1/dense_2/Tensordot/Const_1?
 model_1/dense_2/Tensordot/Prod_1Prod-model_1/dense_2/Tensordot/GatherV2_1:output:0*model_1/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2"
 model_1/dense_2/Tensordot/Prod_1?
%model_1/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2'
%model_1/dense_2/Tensordot/concat/axis?
 model_1/dense_2/Tensordot/concatConcatV2'model_1/dense_2/Tensordot/free:output:0'model_1/dense_2/Tensordot/axes:output:0.model_1/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2"
 model_1/dense_2/Tensordot/concat?
model_1/dense_2/Tensordot/stackPack'model_1/dense_2/Tensordot/Prod:output:0)model_1/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2!
model_1/dense_2/Tensordot/stack?
#model_1/dense_2/Tensordot/transpose	Transpose8model_1/embedding_1/embedding_lookup/Identity_1:output:0)model_1/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2%
#model_1/dense_2/Tensordot/transpose?
!model_1/dense_2/Tensordot/ReshapeReshape'model_1/dense_2/Tensordot/transpose:y:0(model_1/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2#
!model_1/dense_2/Tensordot/Reshape?
 model_1/dense_2/Tensordot/MatMulMatMul*model_1/dense_2/Tensordot/Reshape:output:00model_1/dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2"
 model_1/dense_2/Tensordot/MatMul?
!model_1/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2#
!model_1/dense_2/Tensordot/Const_2?
'model_1/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2)
'model_1/dense_2/Tensordot/concat_1/axis?
"model_1/dense_2/Tensordot/concat_1ConcatV2+model_1/dense_2/Tensordot/GatherV2:output:0*model_1/dense_2/Tensordot/Const_2:output:00model_1/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2$
"model_1/dense_2/Tensordot/concat_1?
model_1/dense_2/TensordotReshape*model_1/dense_2/Tensordot/MatMul:product:0+model_1/dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
model_1/dense_2/Tensordot?
&model_1/dense_2/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02(
&model_1/dense_2/BiasAdd/ReadVariableOp?
model_1/dense_2/BiasAddBiasAdd"model_1/dense_2/Tensordot:output:0.model_1/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
model_1/dense_2/BiasAdd?
model_1/reshape_2/ShapeShape model_1/dense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
model_1/reshape_2/Shape?
%model_1/reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%model_1/reshape_2/strided_slice/stack?
'model_1/reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_2/strided_slice/stack_1?
'model_1/reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'model_1/reshape_2/strided_slice/stack_2?
model_1/reshape_2/strided_sliceStridedSlice model_1/reshape_2/Shape:output:0.model_1/reshape_2/strided_slice/stack:output:00model_1/reshape_2/strided_slice/stack_1:output:00model_1/reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
model_1/reshape_2/strided_slice?
!model_1/reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2#
!model_1/reshape_2/Reshape/shape/1?
!model_1/reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2#
!model_1/reshape_2/Reshape/shape/2?
!model_1/reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/reshape_2/Reshape/shape/3?
model_1/reshape_2/Reshape/shapePack(model_1/reshape_2/strided_slice:output:0*model_1/reshape_2/Reshape/shape/1:output:0*model_1/reshape_2/Reshape/shape/2:output:0*model_1/reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
model_1/reshape_2/Reshape/shape?
model_1/reshape_2/ReshapeReshape model_1/dense_2/BiasAdd:output:0(model_1/reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
model_1/reshape_2/Reshape?
!model_1/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2#
!model_1/concatenate_1/concat/axis?
model_1/concatenate_1/concatConcatV2input_4"model_1/reshape_2/Reshape:output:0*model_1/concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
model_1/concatenate_1/concat?
.model_1/conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp7model_1_conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype020
.model_1/conv_transpose_1/Conv2D/ReadVariableOp?
model_1/conv_transpose_1/Conv2DConv2D%model_1/concatenate_1/concat:output:06model_1/conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2!
model_1/conv_transpose_1/Conv2D?
model_1/leaky_relu_1/LeakyRelu	LeakyRelu(model_1/conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2 
model_1/leaky_relu_1/LeakyRelu?
.model_1/conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp7model_1_conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model_1/conv_transpose_2/Conv2D/ReadVariableOp?
model_1/conv_transpose_2/Conv2DConv2D,model_1/leaky_relu_1/LeakyRelu:activations:06model_1/conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2!
model_1/conv_transpose_2/Conv2D?
model_1/bn_1/ReadVariableOpReadVariableOp$model_1_bn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
model_1/bn_1/ReadVariableOp?
model_1/bn_1/ReadVariableOp_1ReadVariableOp&model_1_bn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model_1/bn_1/ReadVariableOp_1?
,model_1/bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp5model_1_bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_1/bn_1/FusedBatchNormV3/ReadVariableOp?
.model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7model_1_bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1?
model_1/bn_1/FusedBatchNormV3FusedBatchNormV3(model_1/conv_transpose_2/Conv2D:output:0#model_1/bn_1/ReadVariableOp:value:0%model_1/bn_1/ReadVariableOp_1:value:04model_1/bn_1/FusedBatchNormV3/ReadVariableOp:value:06model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
model_1/bn_1/FusedBatchNormV3?
model_1/leaky_relu_2/LeakyRelu	LeakyRelu!model_1/bn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2 
model_1/leaky_relu_2/LeakyRelu?
.model_1/conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp7model_1_conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model_1/conv_transpose_3/Conv2D/ReadVariableOp?
model_1/conv_transpose_3/Conv2DConv2D,model_1/leaky_relu_2/LeakyRelu:activations:06model_1/conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model_1/conv_transpose_3/Conv2D?
model_1/bn_2/ReadVariableOpReadVariableOp$model_1_bn_2_readvariableop_resource*
_output_shapes	
:?*
dtype02
model_1/bn_2/ReadVariableOp?
model_1/bn_2/ReadVariableOp_1ReadVariableOp&model_1_bn_2_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model_1/bn_2/ReadVariableOp_1?
,model_1/bn_2/FusedBatchNormV3/ReadVariableOpReadVariableOp5model_1_bn_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_1/bn_2/FusedBatchNormV3/ReadVariableOp?
.model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7model_1_bn_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1?
model_1/bn_2/FusedBatchNormV3FusedBatchNormV3(model_1/conv_transpose_3/Conv2D:output:0#model_1/bn_2/ReadVariableOp:value:0%model_1/bn_2/ReadVariableOp_1:value:04model_1/bn_2/FusedBatchNormV3/ReadVariableOp:value:06model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
model_1/bn_2/FusedBatchNormV3?
model_1/leaky_relu_3/LeakyRelu	LeakyRelu!model_1/bn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2 
model_1/leaky_relu_3/LeakyRelu?
.model_1/conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp7model_1_conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype020
.model_1/conv_transpose_4/Conv2D/ReadVariableOp?
model_1/conv_transpose_4/Conv2DConv2D,model_1/leaky_relu_3/LeakyRelu:activations:06model_1/conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2!
model_1/conv_transpose_4/Conv2D?
model_1/bn_3/ReadVariableOpReadVariableOp$model_1_bn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
model_1/bn_3/ReadVariableOp?
model_1/bn_3/ReadVariableOp_1ReadVariableOp&model_1_bn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
model_1/bn_3/ReadVariableOp_1?
,model_1/bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp5model_1_bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,model_1/bn_3/FusedBatchNormV3/ReadVariableOp?
.model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp7model_1_bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype020
.model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1?
model_1/bn_3/FusedBatchNormV3FusedBatchNormV3(model_1/conv_transpose_4/Conv2D:output:0#model_1/bn_3/ReadVariableOp:value:0%model_1/bn_3/ReadVariableOp_1:value:04model_1/bn_3/FusedBatchNormV3/ReadVariableOp:value:06model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
model_1/bn_3/FusedBatchNormV3?
model_1/leaky_relu_4/LeakyRelu	LeakyRelu!model_1/bn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2 
model_1/leaky_relu_4/LeakyRelu
model_1/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model_1/flatten/Const?
model_1/flatten/ReshapeReshape,model_1/leaky_relu_4/LeakyRelu:activations:0model_1/flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
model_1/flatten/Reshape?
model_1/dropout/IdentityIdentity model_1/flatten/Reshape:output:0*
T0*)
_output_shapes
:???????????2
model_1/dropout/Identity?
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp?
model_1/dense_3/MatMulMatMul!model_1/dropout/Identity:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/MatMul?
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp?
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/BiasAdd?
model_1/dense_3/SigmoidSigmoid model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
model_1/dense_3/Sigmoidv
IdentityIdentitymodel_1/dense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp-^model_1/bn_1/FusedBatchNormV3/ReadVariableOp/^model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1^model_1/bn_1/ReadVariableOp^model_1/bn_1/ReadVariableOp_1-^model_1/bn_2/FusedBatchNormV3/ReadVariableOp/^model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1^model_1/bn_2/ReadVariableOp^model_1/bn_2/ReadVariableOp_1-^model_1/bn_3/FusedBatchNormV3/ReadVariableOp/^model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1^model_1/bn_3/ReadVariableOp^model_1/bn_3/ReadVariableOp_1/^model_1/conv_transpose_1/Conv2D/ReadVariableOp/^model_1/conv_transpose_2/Conv2D/ReadVariableOp/^model_1/conv_transpose_3/Conv2D/ReadVariableOp/^model_1/conv_transpose_4/Conv2D/ReadVariableOp'^model_1/dense_2/BiasAdd/ReadVariableOp)^model_1/dense_2/Tensordot/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp%^model_1/embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2\
,model_1/bn_1/FusedBatchNormV3/ReadVariableOp,model_1/bn_1/FusedBatchNormV3/ReadVariableOp2`
.model_1/bn_1/FusedBatchNormV3/ReadVariableOp_1.model_1/bn_1/FusedBatchNormV3/ReadVariableOp_12:
model_1/bn_1/ReadVariableOpmodel_1/bn_1/ReadVariableOp2>
model_1/bn_1/ReadVariableOp_1model_1/bn_1/ReadVariableOp_12\
,model_1/bn_2/FusedBatchNormV3/ReadVariableOp,model_1/bn_2/FusedBatchNormV3/ReadVariableOp2`
.model_1/bn_2/FusedBatchNormV3/ReadVariableOp_1.model_1/bn_2/FusedBatchNormV3/ReadVariableOp_12:
model_1/bn_2/ReadVariableOpmodel_1/bn_2/ReadVariableOp2>
model_1/bn_2/ReadVariableOp_1model_1/bn_2/ReadVariableOp_12\
,model_1/bn_3/FusedBatchNormV3/ReadVariableOp,model_1/bn_3/FusedBatchNormV3/ReadVariableOp2`
.model_1/bn_3/FusedBatchNormV3/ReadVariableOp_1.model_1/bn_3/FusedBatchNormV3/ReadVariableOp_12:
model_1/bn_3/ReadVariableOpmodel_1/bn_3/ReadVariableOp2>
model_1/bn_3/ReadVariableOp_1model_1/bn_3/ReadVariableOp_12`
.model_1/conv_transpose_1/Conv2D/ReadVariableOp.model_1/conv_transpose_1/Conv2D/ReadVariableOp2`
.model_1/conv_transpose_2/Conv2D/ReadVariableOp.model_1/conv_transpose_2/Conv2D/ReadVariableOp2`
.model_1/conv_transpose_3/Conv2D/ReadVariableOp.model_1/conv_transpose_3/Conv2D/ReadVariableOp2`
.model_1/conv_transpose_4/Conv2D/ReadVariableOp.model_1/conv_transpose_4/Conv2D/ReadVariableOp2P
&model_1/dense_2/BiasAdd/ReadVariableOp&model_1/dense_2/BiasAdd/ReadVariableOp2T
(model_1/dense_2/Tensordot/ReadVariableOp(model_1/dense_2/Tensordot/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2L
$model_1/embedding_1/embedding_lookup$model_1/embedding_1/embedding_lookup:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
&__inference_bn_2_layer_call_fn_2633851

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
?
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2633591
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
T0*1
_output_shapes
:???????????2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
::???????????:???????????:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:[W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/1
?

?
H__inference_embedding_1_layer_call_and_return_conditional_losses_2633489

inputs*
embedding_lookup_2633483:
identity??embedding_lookup]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:?????????2
Cast?
embedding_lookupResourceGatherembedding_lookup_2633483Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*+
_class!
loc:@embedding_lookup/2633483*+
_output_shapes
:?????????*
dtype02
embedding_lookup?
embedding_lookup/IdentityIdentityembedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*+
_class!
loc:@embedding_lookup/2633483*+
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
?
e
I__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2634119

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
?
2__inference_conv_transpose_4_layer_call_fn_2633958

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_relu_1_layer_call_fn_2633610

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@@?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633923

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
?
?
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2633965

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
2__inference_conv_transpose_2_layer_call_fn_2633622

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633905

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
?
G
+__inference_reshape_2_layer_call_fn_2633563

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
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:???????????2	
Reshapen
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633719

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
?
e
I__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2633783

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????  ?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
D__inference_dense_2_layer_call_and_return_conditional_losses_2633549

inputs5
!tensordot_readvariableop_resource:
??/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
T0*)
_output_shapes
:???????????2
Tensordot/MatMulr
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
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
T0*-
_output_shapes
:???????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2	
BiasAddq
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????2

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
?
?
&__inference_bn_1_layer_call_fn_2633683

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
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633755

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
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633887

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
?
e
I__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2633951

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
?
&__inference_bn_1_layer_call_fn_2633665

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
׏
?
)__inference_model_1_layer_call_fn_2631557
input_4
input_36
$embedding_1_embedding_lookup_2631445:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupv
embedding_1/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2631445embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2631445*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2631445*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_4reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape~
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:???????????2
dropout/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2L
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
2__inference_conv_transpose_1_layer_call_fn_2633598

inputs9
conv2d_readvariableop_resource:?
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????@@?2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:???????????: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633737

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
? 
?
)__inference_dense_2_layer_call_fn_2633519

inputs5
!tensordot_readvariableop_resource:
??/
biasadd_readvariableop_resource:
??
identity??BiasAdd/ReadVariableOp?Tensordot/ReadVariableOp?
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource* 
_output_shapes
:
??*
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
T0*)
_output_shapes
:???????????2
Tensordot/MatMulr
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
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
T0*-
_output_shapes
:???????????2
	Tensordot?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes

:??*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2	
BiasAddq
IdentityIdentityBiasAdd:output:0^NoOp*
T0*-
_output_shapes
:???????????2

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
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634055

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
?
?
&__inference_bn_2_layer_call_fn_2633815

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
?
E
)__inference_flatten_layer_call_fn_2634125

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2633629

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:?????????  ?2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????@@?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2634019

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
?
`
D__inference_flatten_layer_call_and_return_conditional_losses_2634131

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consti
ReshapeReshapeinputsConst:output:0*
T0*)
_output_shapes
:???????????2	
Reshapef
IdentityIdentityReshape:output:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633773

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
?
J
.__inference_leaky_relu_4_layer_call_fn_2634114

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
J
.__inference_leaky_relu_3_layer_call_fn_2633946

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:??????????2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
?
%__inference_signature_wrapper_2632987
input_3
input_4
unknown:
	unknown_0:
??
	unknown_1:
??$
	unknown_2:?%
	unknown_3:??
	unknown_4:	?
	unknown_5:	?
	unknown_6:	?
	unknown_7:	?%
	unknown_8:??
	unknown_9:	?

unknown_10:	?

unknown_11:	?

unknown_12:	?&

unknown_13:??

unknown_14:	?

unknown_15:	?

unknown_16:	?

unknown_17:	?

unknown_18:
??

unknown_19:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_4input_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19*"
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*7
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__wrapped_model_26310012
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:?????????:???????????: : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_3:ZV
1
_output_shapes
:???????????
!
_user_specified_name	input_4
??
?
D__inference_model_1_layer_call_and_return_conditional_losses_2633345
inputs_0
inputs_16
$embedding_1_embedding_lookup_2633233:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2633233embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633233*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633233*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2inputs_0reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_1/FusedBatchNormV3?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_2/FusedBatchNormV3?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%??L?*
is_training( 2
bn_3/FusedBatchNormV3?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshape~
dropout/IdentityIdentityflatten/Reshape:output:0*
T0*)
_output_shapes
:???????????2
dropout/Identity?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/Identity:output:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2L
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
G
)__inference_dropout_layer_call_fn_2634136

inputs

identity_1\
IdentityIdentityinputs*
T0*)
_output_shapes
:???????????2

Identityk

Identity_1IdentityIdentity:output:0*
T0*)
_output_shapes
:???????????2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
&__inference_bn_3_layer_call_fn_2633983

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
??
?
D__inference_model_1_layer_call_and_return_conditional_losses_2633469
inputs_0
inputs_16
$embedding_1_embedding_lookup_2633350:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2633350embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633350*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633350*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2inputs_0reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
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
bn_1/AssignNewValue_1?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
bn_2/AssignNewValue_1?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
bn_3/AssignNewValue_1?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mul_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2*
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634091

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
c
D__inference_dropout_layer_call_and_return_conditional_losses_2634165

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
H
)__inference_dropout_layer_call_fn_2634148

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/Constu
dropout/MulMulinputsdropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/Cast|
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/Mul_1g
IdentityIdentitydropout/Mul_1:z:0*
T0*)
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:???????????:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
J
.__inference_leaky_relu_2_layer_call_fn_2633778

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????  ?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
??
?
)__inference_model_1_layer_call_fn_2632696
input_4
input_36
$embedding_1_embedding_lookup_2632577:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupv
embedding_1/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2632577embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632577*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632577*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_4reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
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
bn_1/AssignNewValue_1?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
bn_2/AssignNewValue_1?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
bn_3/AssignNewValue_1?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mul_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2*
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3
??
?
D__inference_model_1_layer_call_and_return_conditional_losses_2632937
input_4
input_36
$embedding_1_embedding_lookup_2632818:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupv
embedding_1/CastCastinput_3*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2632818embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632818*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2632818*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2input_4reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
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
bn_1/AssignNewValue_1?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
bn_2/AssignNewValue_1?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
bn_3/AssignNewValue_1?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mul_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2*
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:Z V
1
_output_shapes
:???????????
!
_user_specified_name	input_4:PL
'
_output_shapes
:?????????
!
_user_specified_name	input_3
?
?
D__inference_dense_3_layer_call_and_return_conditional_losses_2634187

inputs2
matmul_readvariableop_resource:
??-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Sigmoidf
IdentityIdentitySigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:Q M
)
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633941

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
?
?
&__inference_bn_2_layer_call_fn_2633869

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
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634073

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
&__inference_bn_1_layer_call_fn_2633647

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
?
e
I__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2633615

inputs
identity]
	LeakyRelu	LeakyReluinputs*0
_output_shapes
:?????????@@?2
	LeakyRelut
IdentityIdentityLeakyRelu:activations:0*
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
?
?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634109

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
?
b
F__inference_reshape_2_layer_call_and_return_conditional_losses_2633577

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
strided_slicee
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/1e
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapey
ReshapeReshapeinputsReshape/shape:output:0*
T0*1
_output_shapes
:???????????2	
Reshapen
IdentityIdentityReshape:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*,
_input_shapes
:???????????:U Q
-
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
)__inference_model_1_layer_call_fn_2633228
inputs_0
inputs_16
$embedding_1_embedding_lookup_2633109:=
)dense_2_tensordot_readvariableop_resource:
??7
'dense_2_biasadd_readvariableop_resource:
??J
/conv_transpose_1_conv2d_readvariableop_resource:?K
/conv_transpose_2_conv2d_readvariableop_resource:??+
bn_1_readvariableop_resource:	?-
bn_1_readvariableop_1_resource:	?<
-bn_1_fusedbatchnormv3_readvariableop_resource:	?>
/bn_1_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_3_conv2d_readvariableop_resource:??+
bn_2_readvariableop_resource:	?-
bn_2_readvariableop_1_resource:	?<
-bn_2_fusedbatchnormv3_readvariableop_resource:	?>
/bn_2_fusedbatchnormv3_readvariableop_1_resource:	?K
/conv_transpose_4_conv2d_readvariableop_resource:??+
bn_3_readvariableop_resource:	?-
bn_3_readvariableop_1_resource:	?<
-bn_3_fusedbatchnormv3_readvariableop_resource:	?>
/bn_3_fusedbatchnormv3_readvariableop_1_resource:	?:
&dense_3_matmul_readvariableop_resource:
??5
'dense_3_biasadd_readvariableop_resource:
identity??bn_1/AssignNewValue?bn_1/AssignNewValue_1?$bn_1/FusedBatchNormV3/ReadVariableOp?&bn_1/FusedBatchNormV3/ReadVariableOp_1?bn_1/ReadVariableOp?bn_1/ReadVariableOp_1?bn_2/AssignNewValue?bn_2/AssignNewValue_1?$bn_2/FusedBatchNormV3/ReadVariableOp?&bn_2/FusedBatchNormV3/ReadVariableOp_1?bn_2/ReadVariableOp?bn_2/ReadVariableOp_1?bn_3/AssignNewValue?bn_3/AssignNewValue_1?$bn_3/FusedBatchNormV3/ReadVariableOp?&bn_3/FusedBatchNormV3/ReadVariableOp_1?bn_3/ReadVariableOp?bn_3/ReadVariableOp_1?&conv_transpose_1/Conv2D/ReadVariableOp?&conv_transpose_2/Conv2D/ReadVariableOp?&conv_transpose_3/Conv2D/ReadVariableOp?&conv_transpose_4/Conv2D/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp? dense_2/Tensordot/ReadVariableOp?dense_3/BiasAdd/ReadVariableOp?dense_3/MatMul/ReadVariableOp?embedding_1/embedding_lookupw
embedding_1/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:?????????2
embedding_1/Cast?
embedding_1/embedding_lookupResourceGather$embedding_1_embedding_lookup_2633109embedding_1/Cast:y:0",/job:localhost/replica:0/task:0/device:CPU:0*
Tindices0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633109*+
_output_shapes
:?????????*
dtype02
embedding_1/embedding_lookup?
%embedding_1/embedding_lookup/IdentityIdentity%embedding_1/embedding_lookup:output:0",/job:localhost/replica:0/task:0/device:CPU:0*
T0*7
_class-
+)loc:@embedding_1/embedding_lookup/2633109*+
_output_shapes
:?????????2'
%embedding_1/embedding_lookup/Identity?
'embedding_1/embedding_lookup/Identity_1Identity.embedding_1/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:?????????2)
'embedding_1/embedding_lookup/Identity_1?
 dense_2/Tensordot/ReadVariableOpReadVariableOp)dense_2_tensordot_readvariableop_resource* 
_output_shapes
:
??*
dtype02"
 dense_2/Tensordot/ReadVariableOpz
dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:2
dense_2/Tensordot/axes?
dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       2
dense_2/Tensordot/free?
dense_2/Tensordot/ShapeShape0embedding_1/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
dense_2/Tensordot/Shape?
dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/GatherV2/axis?
dense_2/Tensordot/GatherV2GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/free:output:0(dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2?
!dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2#
!dense_2/Tensordot/GatherV2_1/axis?
dense_2/Tensordot/GatherV2_1GatherV2 dense_2/Tensordot/Shape:output:0dense_2/Tensordot/axes:output:0*dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:2
dense_2/Tensordot/GatherV2_1|
dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const?
dense_2/Tensordot/ProdProd#dense_2/Tensordot/GatherV2:output:0 dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod?
dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 2
dense_2/Tensordot/Const_1?
dense_2/Tensordot/Prod_1Prod%dense_2/Tensordot/GatherV2_1:output:0"dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 2
dense_2/Tensordot/Prod_1?
dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
dense_2/Tensordot/concat/axis?
dense_2/Tensordot/concatConcatV2dense_2/Tensordot/free:output:0dense_2/Tensordot/axes:output:0&dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat?
dense_2/Tensordot/stackPackdense_2/Tensordot/Prod:output:0!dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/stack?
dense_2/Tensordot/transpose	Transpose0embedding_1/embedding_lookup/Identity_1:output:0!dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:?????????2
dense_2/Tensordot/transpose?
dense_2/Tensordot/ReshapeReshapedense_2/Tensordot/transpose:y:0 dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:??????????????????2
dense_2/Tensordot/Reshape?
dense_2/Tensordot/MatMulMatMul"dense_2/Tensordot/Reshape:output:0(dense_2/Tensordot/ReadVariableOp:value:0*
T0*)
_output_shapes
:???????????2
dense_2/Tensordot/MatMul?
dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:??2
dense_2/Tensordot/Const_2?
dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2!
dense_2/Tensordot/concat_1/axis?
dense_2/Tensordot/concat_1ConcatV2#dense_2/Tensordot/GatherV2:output:0"dense_2/Tensordot/Const_2:output:0(dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2
dense_2/Tensordot/concat_1?
dense_2/TensordotReshape"dense_2/Tensordot/MatMul:product:0#dense_2/Tensordot/concat_1:output:0*
T0*-
_output_shapes
:???????????2
dense_2/Tensordot?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes

:??*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/Tensordot:output:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*-
_output_shapes
:???????????2
dense_2/BiasAddj
reshape_2/ShapeShapedense_2/BiasAdd:output:0*
T0*
_output_shapes
:2
reshape_2/Shape?
reshape_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_2/strided_slice/stack?
reshape_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_1?
reshape_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_2/strided_slice/stack_2?
reshape_2/strided_sliceStridedSlicereshape_2/Shape:output:0&reshape_2/strided_slice/stack:output:0(reshape_2/strided_slice/stack_1:output:0(reshape_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_2/strided_slicey
reshape_2/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/1y
reshape_2/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_2/Reshape/shape/2x
reshape_2/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_2/Reshape/shape/3?
reshape_2/Reshape/shapePack reshape_2/strided_slice:output:0"reshape_2/Reshape/shape/1:output:0"reshape_2/Reshape/shape/2:output:0"reshape_2/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_2/Reshape/shape?
reshape_2/ReshapeReshapedense_2/BiasAdd:output:0 reshape_2/Reshape/shape:output:0*
T0*1
_output_shapes
:???????????2
reshape_2/Reshapex
concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate_1/concat/axis?
concatenate_1/concatConcatV2inputs_0reshape_2/Reshape:output:0"concatenate_1/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????2
concatenate_1/concat?
&conv_transpose_1/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_1_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02(
&conv_transpose_1/Conv2D/ReadVariableOp?
conv_transpose_1/Conv2DConv2Dconcatenate_1/concat:output:0.conv_transpose_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@@?*
paddingSAME*
strides
2
conv_transpose_1/Conv2D?
leaky_relu_1/LeakyRelu	LeakyRelu conv_transpose_1/Conv2D:output:0*0
_output_shapes
:?????????@@?2
leaky_relu_1/LeakyRelu?
&conv_transpose_2/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_2/Conv2D/ReadVariableOp?
conv_transpose_2/Conv2DConv2D$leaky_relu_1/LeakyRelu:activations:0.conv_transpose_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????  ?*
paddingSAME*
strides
2
conv_transpose_2/Conv2D?
bn_1/ReadVariableOpReadVariableOpbn_1_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp?
bn_1/ReadVariableOp_1ReadVariableOpbn_1_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_1/ReadVariableOp_1?
$bn_1/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_1/FusedBatchNormV3/ReadVariableOp?
&bn_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_1/FusedBatchNormV3/ReadVariableOp_1?
bn_1/FusedBatchNormV3FusedBatchNormV3 conv_transpose_2/Conv2D:output:0bn_1/ReadVariableOp:value:0bn_1/ReadVariableOp_1:value:0,bn_1/FusedBatchNormV3/ReadVariableOp:value:0.bn_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:?????????  ?:?:?:?:?:*
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
bn_1/AssignNewValue_1?
leaky_relu_2/LeakyRelu	LeakyRelubn_1/FusedBatchNormV3:y:0*0
_output_shapes
:?????????  ?2
leaky_relu_2/LeakyRelu?
&conv_transpose_3/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_3_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_3/Conv2D/ReadVariableOp?
conv_transpose_3/Conv2DConv2D$leaky_relu_2/LeakyRelu:activations:0.conv_transpose_3/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_3/Conv2D?
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
bn_2/FusedBatchNormV3FusedBatchNormV3 conv_transpose_3/Conv2D:output:0bn_2/ReadVariableOp:value:0bn_2/ReadVariableOp_1:value:0,bn_2/FusedBatchNormV3/ReadVariableOp:value:0.bn_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
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
bn_2/AssignNewValue_1?
leaky_relu_3/LeakyRelu	LeakyRelubn_2/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_3/LeakyRelu?
&conv_transpose_4/Conv2D/ReadVariableOpReadVariableOp/conv_transpose_4_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&conv_transpose_4/Conv2D/ReadVariableOp?
conv_transpose_4/Conv2DConv2D$leaky_relu_3/LeakyRelu:activations:0.conv_transpose_4/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv_transpose_4/Conv2D?
bn_3/ReadVariableOpReadVariableOpbn_3_readvariableop_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp?
bn_3/ReadVariableOp_1ReadVariableOpbn_3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02
bn_3/ReadVariableOp_1?
$bn_3/FusedBatchNormV3/ReadVariableOpReadVariableOp-bn_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype02&
$bn_3/FusedBatchNormV3/ReadVariableOp?
&bn_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp/bn_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype02(
&bn_3/FusedBatchNormV3/ReadVariableOp_1?
bn_3/FusedBatchNormV3FusedBatchNormV3 conv_transpose_4/Conv2D:output:0bn_3/ReadVariableOp:value:0bn_3/ReadVariableOp_1:value:0,bn_3/FusedBatchNormV3/ReadVariableOp:value:0.bn_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
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
bn_3/AssignNewValue_1?
leaky_relu_4/LeakyRelu	LeakyRelubn_3/FusedBatchNormV3:y:0*0
_output_shapes
:??????????2
leaky_relu_4/LeakyReluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatten/Const?
flatten/ReshapeReshape$leaky_relu_4/LeakyRelu:activations:0flatten/Const:output:0*
T0*)
_output_shapes
:???????????2
flatten/Reshapes
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
dropout/dropout/Const?
dropout/dropout/MulMulflatten/Reshape:output:0dropout/dropout/Const:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mulv
dropout/dropout/ShapeShapeflatten/Reshape:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*)
_output_shapes
:???????????*
dtype02.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *??L>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*)
_output_shapes
:???????????2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*)
_output_shapes
:???????????2
dropout/dropout/Mul_1?
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype02
dense_3/MatMul/ReadVariableOp?
dense_3/MatMulMatMuldropout/dropout/Mul_1:z:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/MatMul?
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp?
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_3/BiasAddy
dense_3/SigmoidSigmoiddense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
dense_3/Sigmoidn
IdentityIdentitydense_3/Sigmoid:y:0^NoOp*
T0*'
_output_shapes
:?????????2

Identity?
NoOpNoOp^bn_1/AssignNewValue^bn_1/AssignNewValue_1%^bn_1/FusedBatchNormV3/ReadVariableOp'^bn_1/FusedBatchNormV3/ReadVariableOp_1^bn_1/ReadVariableOp^bn_1/ReadVariableOp_1^bn_2/AssignNewValue^bn_2/AssignNewValue_1%^bn_2/FusedBatchNormV3/ReadVariableOp'^bn_2/FusedBatchNormV3/ReadVariableOp_1^bn_2/ReadVariableOp^bn_2/ReadVariableOp_1^bn_3/AssignNewValue^bn_3/AssignNewValue_1%^bn_3/FusedBatchNormV3/ReadVariableOp'^bn_3/FusedBatchNormV3/ReadVariableOp_1^bn_3/ReadVariableOp^bn_3/ReadVariableOp_1'^conv_transpose_1/Conv2D/ReadVariableOp'^conv_transpose_2/Conv2D/ReadVariableOp'^conv_transpose_3/Conv2D/ReadVariableOp'^conv_transpose_4/Conv2D/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp!^dense_2/Tensordot/ReadVariableOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^embedding_1/embedding_lookup*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*m
_input_shapes\
Z:???????????:?????????: : : : : : : : : : : : : : : : : : : : : 2*
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
bn_3/ReadVariableOp_1bn_3/ReadVariableOp_12P
&conv_transpose_1/Conv2D/ReadVariableOp&conv_transpose_1/Conv2D/ReadVariableOp2P
&conv_transpose_2/Conv2D/ReadVariableOp&conv_transpose_2/Conv2D/ReadVariableOp2P
&conv_transpose_3/Conv2D/ReadVariableOp&conv_transpose_3/Conv2D/ReadVariableOp2P
&conv_transpose_4/Conv2D/ReadVariableOp&conv_transpose_4/Conv2D/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2D
 dense_2/Tensordot/ReadVariableOp dense_2/Tensordot/ReadVariableOp2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2<
embedding_1/embedding_lookupembedding_1/embedding_lookup:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:?????????
"
_user_specified_name
inputs/1
?
?
&__inference_bn_1_layer_call_fn_2633701

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
?
?
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2633797

inputs:
conv2d_readvariableop_resource:??
identity??Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2Ds
IdentityIdentityConv2D:output:0^NoOp*
T0*0
_output_shapes
:??????????2

Identityf
NoOpNoOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:?????????  ?: 2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????  ?
 
_user_specified_nameinputs"?-
saver_filename:0
Identity:0Identity_228"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_30
serving_default_input_3:0?????????
E
input_4:
serving_default_input_4:0???????????;
dense_30
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer_with_weights-4

layer-9
layer-10
layer_with_weights-5
layer-11
layer_with_weights-6
layer-12
layer-13
layer_with_weights-7
layer-14
layer_with_weights-8
layer-15
layer-16
layer-17
layer-18
layer_with_weights-9
layer-19

signatures
#_self_saveable_object_factories
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"
_tf_keras_network
D
#_self_saveable_object_factories"
_tf_keras_input_layer
?

embeddings
#_self_saveable_object_factories
trainable_variables
	variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

"kernel
#bias
#$_self_saveable_object_factories
%trainable_variables
&	variables
'regularization_losses
(	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
D
#)_self_saveable_object_factories"
_tf_keras_input_layer
?
#*_self_saveable_object_factories
+trainable_variables
,	variables
-regularization_losses
.	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#/_self_saveable_object_factories
0trainable_variables
1	variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

4kernel
#5_self_saveable_object_factories
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#:_self_saveable_object_factories
;trainable_variables
<	variables
=regularization_losses
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

?kernel
#@_self_saveable_object_factories
Atrainable_variables
B	variables
Cregularization_losses
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Eaxis
	Fgamma
Gbeta
Hmoving_mean
Imoving_variance
#J_self_saveable_object_factories
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#O_self_saveable_object_factories
Ptrainable_variables
Q	variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

Tkernel
#U_self_saveable_object_factories
Vtrainable_variables
W	variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
#__self_saveable_object_factories
`trainable_variables
a	variables
bregularization_losses
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#d_self_saveable_object_factories
etrainable_variables
f	variables
gregularization_losses
h	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?

ikernel
#j_self_saveable_object_factories
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
oaxis
	pgamma
qbeta
rmoving_mean
smoving_variance
#t_self_saveable_object_factories
utrainable_variables
v	variables
wregularization_losses
x	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#y_self_saveable_object_factories
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
?__call__
+?&call_and_return_all_conditional_losses"
_tf_keras_layer
?
#~_self_saveable_object_factories
trainable_variables
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
	?bias
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
0
"1
#2
43
?4
F5
G6
T7
[8
\9
i10
p11
q12
?13
?14"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
"1
#2
43
?4
F5
G6
H7
I8
T9
[10
\11
]12
^13
i14
p15
q16
r17
s18
?19
?20"
trackable_list_wrapper
?
trainable_variables
?layer_metrics
 ?layer_regularization_losses
?non_trainable_variables
regularization_losses
?layers
?metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
(:&2embedding_1/embeddings
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
trainable_variables
?layer_metrics
	variables
 ?layer_regularization_losses
 regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
": 
??2dense_2/kernel
:??2dense_2/bias
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
%trainable_variables
?layer_metrics
&	variables
 ?layer_regularization_losses
'regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
+trainable_variables
?layer_metrics
,	variables
 ?layer_regularization_losses
-regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
0trainable_variables
?layer_metrics
1	variables
 ?layer_regularization_losses
2regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0?2conv_transpose_1/kernel
 "
trackable_dict_wrapper
'
40"
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
?
6trainable_variables
?layer_metrics
7	variables
 ?layer_regularization_losses
8regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
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
;trainable_variables
?layer_metrics
<	variables
 ?layer_regularization_losses
=regularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_2/kernel
 "
trackable_dict_wrapper
'
?0"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Atrainable_variables
?layer_metrics
B	variables
 ?layer_regularization_losses
Cregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
:?2
bn_1/gamma
:?2	bn_1/beta
!:? (2bn_1/moving_mean
%:#? (2bn_1/moving_variance
 "
trackable_dict_wrapper
.
F0
G1"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ktrainable_variables
?layer_metrics
L	variables
 ?layer_regularization_losses
Mregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
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
Ptrainable_variables
?layer_metrics
Q	variables
 ?layer_regularization_losses
Rregularization_losses
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
T0"
trackable_list_wrapper
'
T0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Vtrainable_variables
?layer_metrics
W	variables
 ?layer_regularization_losses
Xregularization_losses
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
[0
\1"
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
`trainable_variables
?layer_metrics
a	variables
 ?layer_regularization_losses
bregularization_losses
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
etrainable_variables
?layer_metrics
f	variables
 ?layer_regularization_losses
gregularization_losses
?layers
?metrics
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv_transpose_4/kernel
 "
trackable_dict_wrapper
'
i0"
trackable_list_wrapper
'
i0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
ktrainable_variables
?layer_metrics
l	variables
 ?layer_regularization_losses
mregularization_losses
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
bn_3/gamma
:?2	bn_3/beta
!:? (2bn_3/moving_mean
%:#? (2bn_3/moving_variance
 "
trackable_dict_wrapper
.
p0
q1"
trackable_list_wrapper
<
p0
q1
r2
s3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
utrainable_variables
?layer_metrics
v	variables
 ?layer_regularization_losses
wregularization_losses
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
ztrainable_variables
?layer_metrics
{	variables
 ?layer_regularization_losses
|regularization_losses
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
trainable_variables
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
": 
??2dense_3/kernel
:2dense_3/bias
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
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
J
H0
I1
]2
^3
r4
s5"
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
19"
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
H0
I1"
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
]0
^1"
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
r0
s1"
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
?2?
)__inference_model_1_layer_call_fn_2631557
)__inference_model_1_layer_call_fn_2633104
)__inference_model_1_layer_call_fn_2633228
)__inference_model_1_layer_call_fn_2632696?
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
"__inference__wrapped_model_2631001input_4input_3"?
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
D__inference_model_1_layer_call_and_return_conditional_losses_2633345
D__inference_model_1_layer_call_and_return_conditional_losses_2633469
D__inference_model_1_layer_call_and_return_conditional_losses_2632813
D__inference_model_1_layer_call_and_return_conditional_losses_2632937?
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
-__inference_embedding_1_layer_call_fn_2633479?
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
H__inference_embedding_1_layer_call_and_return_conditional_losses_2633489?
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
)__inference_dense_2_layer_call_fn_2633519?
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
D__inference_dense_2_layer_call_and_return_conditional_losses_2633549?
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
+__inference_reshape_2_layer_call_fn_2633563?
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
F__inference_reshape_2_layer_call_and_return_conditional_losses_2633577?
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
/__inference_concatenate_1_layer_call_fn_2633584?
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
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2633591?
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
2__inference_conv_transpose_1_layer_call_fn_2633598?
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
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2633605?
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
.__inference_leaky_relu_1_layer_call_fn_2633610?
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
I__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2633615?
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
2__inference_conv_transpose_2_layer_call_fn_2633622?
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
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2633629?
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
&__inference_bn_1_layer_call_fn_2633647
&__inference_bn_1_layer_call_fn_2633665
&__inference_bn_1_layer_call_fn_2633683
&__inference_bn_1_layer_call_fn_2633701?
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
A__inference_bn_1_layer_call_and_return_conditional_losses_2633719
A__inference_bn_1_layer_call_and_return_conditional_losses_2633737
A__inference_bn_1_layer_call_and_return_conditional_losses_2633755
A__inference_bn_1_layer_call_and_return_conditional_losses_2633773?
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
.__inference_leaky_relu_2_layer_call_fn_2633778?
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
I__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2633783?
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
2__inference_conv_transpose_3_layer_call_fn_2633790?
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
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2633797?
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
&__inference_bn_2_layer_call_fn_2633815
&__inference_bn_2_layer_call_fn_2633833
&__inference_bn_2_layer_call_fn_2633851
&__inference_bn_2_layer_call_fn_2633869?
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
A__inference_bn_2_layer_call_and_return_conditional_losses_2633887
A__inference_bn_2_layer_call_and_return_conditional_losses_2633905
A__inference_bn_2_layer_call_and_return_conditional_losses_2633923
A__inference_bn_2_layer_call_and_return_conditional_losses_2633941?
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
.__inference_leaky_relu_3_layer_call_fn_2633946?
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
I__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2633951?
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
2__inference_conv_transpose_4_layer_call_fn_2633958?
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
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2633965?
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
&__inference_bn_3_layer_call_fn_2633983
&__inference_bn_3_layer_call_fn_2634001
&__inference_bn_3_layer_call_fn_2634019
&__inference_bn_3_layer_call_fn_2634037?
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
A__inference_bn_3_layer_call_and_return_conditional_losses_2634055
A__inference_bn_3_layer_call_and_return_conditional_losses_2634073
A__inference_bn_3_layer_call_and_return_conditional_losses_2634091
A__inference_bn_3_layer_call_and_return_conditional_losses_2634109?
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
.__inference_leaky_relu_4_layer_call_fn_2634114?
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
I__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2634119?
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
)__inference_flatten_layer_call_fn_2634125?
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
D__inference_flatten_layer_call_and_return_conditional_losses_2634131?
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
)__inference_dropout_layer_call_fn_2634136
)__inference_dropout_layer_call_fn_2634148?
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
?2?
D__inference_dropout_layer_call_and_return_conditional_losses_2634153
D__inference_dropout_layer_call_and_return_conditional_losses_2634165?
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
)__inference_dense_3_layer_call_fn_2634176?
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
D__inference_dense_3_layer_call_and_return_conditional_losses_2634187?
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
%__inference_signature_wrapper_2632987input_3input_4"?
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
"__inference__wrapped_model_2631001?"#4?FGHIT[\]^ipqrs??b?_
X?U
S?P
+?(
input_4???????????
!?
input_3?????????
? "1?.
,
dense_3!?
dense_3??????????
A__inference_bn_1_layer_call_and_return_conditional_losses_2633719?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633737?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633755tFGHI<?9
2?/
)?&
inputs?????????  ?
p 
? ".?+
$?!
0?????????  ?
? ?
A__inference_bn_1_layer_call_and_return_conditional_losses_2633773tFGHI<?9
2?/
)?&
inputs?????????  ?
p
? ".?+
$?!
0?????????  ?
? ?
&__inference_bn_1_layer_call_fn_2633647?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_1_layer_call_fn_2633665?FGHIN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_1_layer_call_fn_2633683gFGHI<?9
2?/
)?&
inputs?????????  ?
p 
? "!??????????  ??
&__inference_bn_1_layer_call_fn_2633701gFGHI<?9
2?/
)?&
inputs?????????  ?
p
? "!??????????  ??
A__inference_bn_2_layer_call_and_return_conditional_losses_2633887?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633905?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633923t[\]^<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_bn_2_layer_call_and_return_conditional_losses_2633941t[\]^<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_bn_2_layer_call_fn_2633815?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_2_layer_call_fn_2633833?[\]^N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_2_layer_call_fn_2633851g[\]^<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_bn_2_layer_call_fn_2633869g[\]^<?9
2?/
)?&
inputs??????????
p
? "!????????????
A__inference_bn_3_layer_call_and_return_conditional_losses_2634055?pqrsN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634073?pqrsN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634091tpqrs<?9
2?/
)?&
inputs??????????
p 
? ".?+
$?!
0??????????
? ?
A__inference_bn_3_layer_call_and_return_conditional_losses_2634109tpqrs<?9
2?/
)?&
inputs??????????
p
? ".?+
$?!
0??????????
? ?
&__inference_bn_3_layer_call_fn_2633983?pqrsN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
&__inference_bn_3_layer_call_fn_2634001?pqrsN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
&__inference_bn_3_layer_call_fn_2634019gpqrs<?9
2?/
)?&
inputs??????????
p 
? "!????????????
&__inference_bn_3_layer_call_fn_2634037gpqrs<?9
2?/
)?&
inputs??????????
p
? "!????????????
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2633591?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? "/?,
%?"
0???????????
? ?
/__inference_concatenate_1_layer_call_fn_2633584?n?k
d?a
_?\
,?)
inputs/0???????????
,?)
inputs/1???????????
? ""?????????????
M__inference_conv_transpose_1_layer_call_and_return_conditional_losses_2633605n49?6
/?,
*?'
inputs???????????
? ".?+
$?!
0?????????@@?
? ?
2__inference_conv_transpose_1_layer_call_fn_2633598a49?6
/?,
*?'
inputs???????????
? "!??????????@@??
M__inference_conv_transpose_2_layer_call_and_return_conditional_losses_2633629m?8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????  ?
? ?
2__inference_conv_transpose_2_layer_call_fn_2633622`?8?5
.?+
)?&
inputs?????????@@?
? "!??????????  ??
M__inference_conv_transpose_3_layer_call_and_return_conditional_losses_2633797mT8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0??????????
? ?
2__inference_conv_transpose_3_layer_call_fn_2633790`T8?5
.?+
)?&
inputs?????????  ?
? "!????????????
M__inference_conv_transpose_4_layer_call_and_return_conditional_losses_2633965mi8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
2__inference_conv_transpose_4_layer_call_fn_2633958`i8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_dense_2_layer_call_and_return_conditional_losses_2633549f"#3?0
)?&
$?!
inputs?????????
? "+?(
!?
0???????????
? ?
)__inference_dense_2_layer_call_fn_2633519Y"#3?0
)?&
$?!
inputs?????????
? "?????????????
D__inference_dense_3_layer_call_and_return_conditional_losses_2634187`??1?.
'?$
"?
inputs???????????
? "%?"
?
0?????????
? ?
)__inference_dense_3_layer_call_fn_2634176S??1?.
'?$
"?
inputs???????????
? "???????????
D__inference_dropout_layer_call_and_return_conditional_losses_2634153`5?2
+?(
"?
inputs???????????
p 
? "'?$
?
0???????????
? ?
D__inference_dropout_layer_call_and_return_conditional_losses_2634165`5?2
+?(
"?
inputs???????????
p
? "'?$
?
0???????????
? ?
)__inference_dropout_layer_call_fn_2634136S5?2
+?(
"?
inputs???????????
p 
? "?????????????
)__inference_dropout_layer_call_fn_2634148S5?2
+?(
"?
inputs???????????
p
? "?????????????
H__inference_embedding_1_layer_call_and_return_conditional_losses_2633489_/?,
%?"
 ?
inputs?????????
? ")?&
?
0?????????
? ?
-__inference_embedding_1_layer_call_fn_2633479R/?,
%?"
 ?
inputs?????????
? "???????????
D__inference_flatten_layer_call_and_return_conditional_losses_2634131c8?5
.?+
)?&
inputs??????????
? "'?$
?
0???????????
? ?
)__inference_flatten_layer_call_fn_2634125V8?5
.?+
)?&
inputs??????????
? "?????????????
I__inference_leaky_relu_1_layer_call_and_return_conditional_losses_2633615j8?5
.?+
)?&
inputs?????????@@?
? ".?+
$?!
0?????????@@?
? ?
.__inference_leaky_relu_1_layer_call_fn_2633610]8?5
.?+
)?&
inputs?????????@@?
? "!??????????@@??
I__inference_leaky_relu_2_layer_call_and_return_conditional_losses_2633783j8?5
.?+
)?&
inputs?????????  ?
? ".?+
$?!
0?????????  ?
? ?
.__inference_leaky_relu_2_layer_call_fn_2633778]8?5
.?+
)?&
inputs?????????  ?
? "!??????????  ??
I__inference_leaky_relu_3_layer_call_and_return_conditional_losses_2633951j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_leaky_relu_3_layer_call_fn_2633946]8?5
.?+
)?&
inputs??????????
? "!????????????
I__inference_leaky_relu_4_layer_call_and_return_conditional_losses_2634119j8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
.__inference_leaky_relu_4_layer_call_fn_2634114]8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_model_1_layer_call_and_return_conditional_losses_2632813?"#4?FGHIT[\]^ipqrs??j?g
`?]
S?P
+?(
input_4???????????
!?
input_3?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2632937?"#4?FGHIT[\]^ipqrs??j?g
`?]
S?P
+?(
input_4???????????
!?
input_3?????????
p

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2633345?"#4?FGHIT[\]^ipqrs??l?i
b?_
U?R
,?)
inputs/0???????????
"?
inputs/1?????????
p 

 
? "%?"
?
0?????????
? ?
D__inference_model_1_layer_call_and_return_conditional_losses_2633469?"#4?FGHIT[\]^ipqrs??l?i
b?_
U?R
,?)
inputs/0???????????
"?
inputs/1?????????
p

 
? "%?"
?
0?????????
? ?
)__inference_model_1_layer_call_fn_2631557?"#4?FGHIT[\]^ipqrs??j?g
`?]
S?P
+?(
input_4???????????
!?
input_3?????????
p 

 
? "???????????
)__inference_model_1_layer_call_fn_2632696?"#4?FGHIT[\]^ipqrs??j?g
`?]
S?P
+?(
input_4???????????
!?
input_3?????????
p

 
? "???????????
)__inference_model_1_layer_call_fn_2633104?"#4?FGHIT[\]^ipqrs??l?i
b?_
U?R
,?)
inputs/0???????????
"?
inputs/1?????????
p 

 
? "???????????
)__inference_model_1_layer_call_fn_2633228?"#4?FGHIT[\]^ipqrs??l?i
b?_
U?R
,?)
inputs/0???????????
"?
inputs/1?????????
p

 
? "???????????
F__inference_reshape_2_layer_call_and_return_conditional_losses_2633577h5?2
+?(
&?#
inputs???????????
? "/?,
%?"
0???????????
? ?
+__inference_reshape_2_layer_call_fn_2633563[5?2
+?(
&?#
inputs???????????
? ""?????????????
%__inference_signature_wrapper_2632987?"#4?FGHIT[\]^ipqrs??s?p
? 
i?f
,
input_3!?
input_3?????????
6
input_4+?(
input_4???????????"1?.
,
dense_3!?
dense_3?????????