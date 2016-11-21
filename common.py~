import numpy as np
import os

# classes dictionary: from string to float
cdict={'A':0,'E':1,'N':2,'P':3,'R':4}

class Data:
	''' here we load all kinds of data: lld MMFC...

	Attention:
	the data path: ./Data/features_labels_#type#

	Attributes:
	type: lld MMFC
 
	feature:samplesxdimension float training data 
	label:samplesx[classes probility] float  traning data

	feature_test: same as feature. test data
	label_test: same sa label. test data
	
	'''
	#def __init__(self,tp):
		#print tp

	def __init__(self, tp):
		self.type=tp
		#print tp
		#load_data()
		
	def load_data(self):
		''' load data from files
		'''
		print self.type
		if self.type=='lld':
			self.feature=self.load_lld('./Data/features_labels_lld/lld/train/')
			self.feature_test=self.load_lld('./Data/features_labels_lld/lld/test/')
			self.label=self.load_label('./Data/features_labels_lld/labels/train.txt')
			self.label_test=self.load_label('./Data/features_labels_lld/labels/test.txt')
		else:
			# TODO:other features
			pass

	def load_lld(self,lld_dir):
		name_list=os.listdir(lld_dir)
		feature=np.zeros((len(name_list),384))
		for index,name in enumerate(name_list):
			lld_name=lld_dir+name
			lld=np.loadtxt(open(lld_name,'rb'),delimiter=',')
			feature[index,:]=lld.T

		return feature

	def load_label(self,label_file):
		lines=open(label_file,'rb').readlines()
		label=np.zeros((len(lines),2))
		for index,line in enumerate(lines):
			#print line.strip().split
			[_,c,p]=line.strip().split(' ')
			[label[index,0],label[index,1]]=[cdict[c],np.float(p)]
		
		return label
	
