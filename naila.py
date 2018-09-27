import yaml  
import sys
import random
import nltk
import operator
import jellyfish as jf
import json
import requests
import os
import time
import signal
import subprocess
from nltk.tag import StanfordPOSTagger
from textblob.classifiers import NaiveBayesClassifier
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn import preprocessing
from nltk.corpus import wordnet 


#nltk.download('punkt')
def signal_handler(signal, frame):
    print ('Thank You!')
    sys.exit(0)


#mylist= ["file", "folder", "directory", "contents", "name","document"]

data={
    'transit': {
        
        'mv':["move","relocate", "rename", "replace", "send"],
        'cp':["copy","replicate","duplicate"],
        'rm':["remove","delete", "send", "trash"],
	
    },
    'change': {
        'chgrp':["group"],
        'chmod':["permission","access", "mode"],
        'chown':["ownership"],
        'passwd':["password"],
	'cd':["change", "directory", "go", "take", "navigate","to"],
    },
    'display': {
        'ls':["list","directory","folder","display","print","show","enlist"],
        'cat':["concatenate","combine","print","display","show","file", "document", "content"],
        'dirname':["name"],
        'echo':["text","show","echo","screen"],
        'less':["less"],
        'more':["more"],
		'gedit':["open", "file", "document"],
        'head':["first","lines", "display", "starting"],
        'tail':["last","lines", "display", "ending"],
        'man':["manual","help"],
        'ps':["process","active","running","current","snapshot","print","show"],
        'who':["logged","user","username","id"],
        'whoami':["current","user","username","id"],
        'cal':["display","print","show","calendar"],
        'date':["what","today","date","display","print","show"],
        'pwd':["working","directory","current", "print", "show", "display"],
        'ifconfig':["ip","address","show","display","print","show","configure","network","interface"],
		'du':["size", "folder", "space", "taken", "current"],
		'free':["free", "space", "available", "memory","used", "system", "pc"],
		'history':["previous", "history", "command", "terminal", "display"],
    },
    'create': {
        'mkdir':[ "directory", "folder"],
        'mkfifo':["named", "pipe"],
        'mknod':["special","file",],
        'touch':["file","update", "timestamp", "document"],
        'ln':["symbolic", "hard", "link"],
        
    },
    'compare': {
        'cmp':[ "byte", "files", "compare", "document"],
        'diff':["files","compare", "document"],
        
    },
    'search': {
        'grep':[ "match", "regular", "expression", "pattern"],
	'find':[ "file", "document", "search", "find"],
        
    },
    'system': {
	'clear':["clean", "terminal"],
	'reboot':["restart", "reboot", "system"],
	'shutdown':["switch", "off", "switch off", "close", "shut", "down", "power", "shutdown", "poweroff"],
	'sleep':["sleep", "temporary", "close"],
	'gzip': ["compress", "file", "zip"],
	'gunzip': ["decompress", "uncompress", "unzip", "extract", "file"],
	'tar -czvf':["compress", "zip", "folder"],
	'uncompress':["decompress", "uncompress", "unzip", "extract", "folder"],
	
    },
}


command_format={
	"ls":["noun"],
	"mv":["noun", "noun"],
	"cp":["noun", "noun"],
	"rm":["noun", "noun"],
	"cat":["noun", "noun"],
	"dirname":["string"],
	"echo":["string"],
	"more":["noun"],
	"less":["noun"],
	"head":["cardinal", "noun"],
	"tail":["cardinal","noun"],
	"man":["noun"],
	"ps":[],
	"who":[],
	"whoami":[],
	"cal":[],
	"date":[],
	"pwd":[],
	"ifconfig":[],
	"mkdir":["noun"],
	"mkfifo":["noun"],
	"mknod":["noun"],
	"touch":["noun"],
	"ln":["noun", "noun"],
	"cmp":["noun", "noun"],
	"diff":["noun", "noun"],
	"grep":["noun", "noun"],
	"chgrp":["noun", "noun"],
	"chmod":[],
	"chown":[],
	"passwd":[],
	"find":["noun"],
	"gedit":["noun"],
	"du":["noun"],
	"clear":[],
	"reboot":[],
	"shutdown":[],
	"sleep":["noun"],
	"free":[],
	"gzip": ["noun"],
	"gunzip": ["noun"],
	"tar -czvf":["noun"],
	"uncompress":["noun"],
	"cd":["noun"],
	"history":[],
}

description={
	"mv": "Move, Rename or Replace a file ",
	"cp": "Copy a file",
	"rm": "Remove or Delete an item",
	"passwd": "Change user password",
	"ls": "list directory contents",
	"cat": "Concatenate files and print on standard output or show contents of a file",
	"dirname": "strip last component from file name",
	"echo": "display a line of text",
	"less": "show less of a file",
	"more": "show more of a file",
	"gedit": "open a file in text editor",
	"head": "show starting lines of a file",
	"tail": "show ending lines of a text",
	"man": "manual/ help for a command",
	"ps": "show running processes",
	"who": "Print the current username",
	"whoami": "Print the hostname of the system",
	"cal": "Show calendar",
	"date": "Show date",
	"pwd": "Show current working directory",
	"ifconfig": "Show network configuration",
	"mkdir": "Create a new directory ",
	"mkfifo": "make names pipes",
	"mknod": "make a special file",
	"touch": "Create a file/ Update timestamp of existing file",
	"ln": "make links between files",
	"cmp": "Compare two files byte by byte",
	"diff": "Compare two files",
	"grep": "Search a pattern",
	"find": "Find a file in pc",
	"du":"Display the size of the directory",
	"clear":"clean the terminal",
	"reboot":"restart the system",
	"shutdown":"power off the system",
	"sleep":"sleep the system for an interval of time",
	"free":"show the free and used memory of the system",
	"gzip":"compress a file",
	"gunzip":"uncompress a file",
	"tar -czvf":"compress a folder",
	"uncompress":"uncompress a folder",
	"chmod":"changes the permissions of file or folder command format is:\nchmod options permissions filename Example:\n\t1)chmod u=rwx,g=rx,o=r myfile\n\t2)chmod 754 myfile\n",
	"chgrp":"changes the group ownership of a file/files",
	"chown":"changes the ownership of files and directories in Linux. There are three types of file permissions, User, Group and Other",
	"cd": "Change to the specified directory",
	"history":"Show history of commands",

}




noun=[]
adjective=[]
verb=[]
adverb=[]
determiner=[]
pronoun=[]
modal=[]
particle=[]
symbol=[]
cardinal=[]
conjunction=[]
preposition=[]
interjection=[]
existential=[]


    

signal.signal(signal.SIGINT, signal_handler)

my_path = os.path.abspath(os.path.dirname(__file__))

CONFIG_PATH = os.path.join(my_path, "./config.yml")
TRAINDATA_PATH = os.path.join(my_path, "./traindata2.txt")
LABEL_PATH = os.path.join(my_path, "./")

sys.path.insert(0, LABEL_PATH)
import trainlabel2

with open(CONFIG_PATH,"r") as config_file:
    config = yaml.load(config_file)

os.environ['STANFORD_MODELS'] = config['tagger']['path_to_models']

exec_command = config['preferences']['execute']

    
def classify(text):
    X_train = np.array([line.rstrip('\n') for line in open(TRAINDATA_PATH)])
    y_train_text = trainlabel2.y_train_text
    #print y_train_text
    #print X_train
    X_test = np.array([text])
    #target_names = ['file', 'folder', 'network', 'system', 'general']
    
    lb = preprocessing.MultiLabelBinarizer()
    Y = lb.fit_transform(y_train_text)
    #print Y
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])
   # classifier=OneVsRestClassifier(LinearSVC())
    classifier.fit(X_train, Y)
    predicted = classifier.predict(X_test)
    all_labels = lb.inverse_transform(predicted)
    
    for item, labels in zip(X_test, all_labels):
        return (', '.join(labels))

def suggestions(suggest_list):
    suggest = (sorted(suggest_list,reverse=True)[:5])
    return suggest

def execute_command(command):
	import subprocess
	p = subprocess.Popen([command], stdout=subprocess.PIPE, shell=True)
	output, err = p.communicate()
	print  output

def find_arg(category,stanford_tag):
	for item in stanford_tag:
		if item[1]=="VB" or item[1]=="VBD" or item[1]=="VBG" or item[1]=="VBN" or item[1]=="VBP" or item[1]=="VBZ":
	 		verb.append(item[0])
		elif item[1]=="NN" or item[1]=="NNS" or item[1]=="NNP" or item[1]=="NNPS":
			noun.append(item[0])
		elif item[1]=="JJ" or item[1]=="JJR" or item[1]=="JJS":
			adjective.append(item[0])
		elif item[1]=="RB" or item[1]=="RBR" or item[1]=="RBS":
			adverb.append(item[0])
		elif item[1]=="WDT" or item[1]=="PDT" or item[1]=="DT":
			determiner.append(item[0])
		elif item[1]=="PRP" or item[1]=="PRP$" or item[1]=="WP$":
	    		pronoun.append(item[0])
	        elif item[1]=="MD":
	    		modal.append(item[0])
	        elif item[1]=="RP":
	    		particle.appen(item[0])
       	        elif item[1]=="SYM":
	    		symbol.append(item[0])
	        elif item[1]=="CD":
	    		cardinal.append(item[0])
	        elif item[1]=="CC":
			conjunction.append(item[0])
		elif item[1]=="IN":
			preposition.append(item[0])
		elif item[1]=="UH":
			interjection.append(item[0])
		elif item[1]=="EX":
			existential.append(item[0])

		


	"""print"verbs are"
	for item in verb:
		print item
	print "nouns are"
	for item in noun:
		print item"""


def construct_command(category):
	myformat=command_format[category]
	mycommand=""
	mycommand= mycommand+ str(category)+ str(" ")
	if category=="chgrp":
		if len(noun)>=2:
			mycommand= mycommand+ str(noun[1]) +str(" ")+ str(noun[2])
	elif category=="tar -czvf":
		if len(noun)>=1:
			filename= ""
			filename= noun[0]+".tar.gz"
			mycommand= mycommand+ filename +str(" ")+ str(noun[0])

	elif category=="head" or category=="tail":
		if len(noun)>=1 and len(cardinal)>=1:
			mycommand= mycommand+ " -n "+ str(cardinal[0]) +str(" ")+ str(noun[0])
		elif len(noun)>=1:
			mycommand= mycommand+ str(noun[0])
	else:
		for item in myformat:
			if item=="noun":
				if len(noun)!=0 :
					mycommand=mycommand+str(noun[0])+str(" ")
					del noun[0]
				else:
					continue
			elif item=="verb":
				mycommand=mycommand+str(verb[0])+str(" ")
				del verb[0]
	
	print mycommand
	
	print "executing..."
	execute_command(mycommand)


#first_arg = sys.argv[1]
first_arg=' '.join(sys.argv[1:])

def call_reia(user_input=first_arg):
    		max_score = 0.1
		
		
		#print('-----------------------')
		#user_name = get_username(first_line.split(' ', 1)[0])
		suggest_list = []
		suggest_message = ""
		#prev_ts = ts
		#print("\nINPUT = ")
		#print(user_input)
		label = classify(user_input)
		if label == "":
			print("Sorry, I could not understand. Please rephrase and try again.")
			sys.exit()
			#consume_message()
			
		#print("Classified as : "+str(label))
		
		cnt=0;
		
			
		#print(stanford_tag)
		tokens = nltk.word_tokenize(user_input)
		#print(tokens)
		sentence_tokens= []
		for i in tokens:
			if i == label:
				continue
			w1 = wordnet.synsets(i)
			w2 = wordnet.synsets(label)
			flag=False
			for item1 in w1: 
				for item2 in w2:
					dist=item1.path_similarity(item2)
					#dist2=item1.wup_similarity(item2)
					#print(item1, " ", item2, " ", dist)
					if dist>=0.8:
						#print(item1,"#")
						#print(item1.wup_similarity(item2)," ",label," ",i)
						flag=True
			if flag:
				continue
			sentence_tokens.append(i)
		#print "sent token"
		#for item in sentence_tokens:
		#	print item

		#with open(MAPPING_PATH,'r') as data_file:    
		#	data = json.load(data_file)





		maxlabel=0
		#maxcnt=0
		category=""
		"""		
		for comm in data[label]:
			print comm
			for item in data[label][comm]:
				print item
		"""
		posscomm = []
		f=1
		for comm in data[label]:
			cnt=0
			#maxcnt=0
			thresh =0
			#print comm
			#print "::"
			for item in data[label][comm]:
				for i in sentence_tokens:
					##################################3
					#dist = jf.jaro_distance(unicode((item),encoding="utf-8"), unicode(str(i),encoding="utf-8"))
					thresh=0
					#print(item, " ", i)
					#print
					w1 = wordnet.synsets(item)
					w2 = wordnet.synsets(str(i))
					flag=False
					for item1 in w1:
						for item2 in w2:
							dist=item1.path_similarity(item2)
							
							if dist>=0.8:
								#print "***"
								#print
								#print(item1.wup_similarity(item2)," ",item," ",i)
								flag=True

								thresh=thresh+1

					if flag:
						cnt=cnt+1
						#print "i "+i+"item "+item

			if (cnt>=maxlabel and cnt!=0):
				posscomm.append(comm)
				#print "possible command is:"+comm
				maxlabel=cnt
				category=comm
		if(len(posscomm)==1):
			print "category is:"+category
			print
			f= input("Would you like to execute the above command? \n Enter 1 for yes and 0 for no ")
		else:
			counter=1
			print "Please choose the appropriate command from the following list of possible commands:"
			for j in posscomm:
				print counter, " - ",j, " : ", description[j]
				counter=counter+1

			print counter,"Abort"
			if category=="chmod" or category=="chgrp":
				exit()
		
			choice= input("Enter the choice: ")
		
			if choice==counter:
				exit()
			
			category= posscomm[choice-1]
			#print "category is:"+category

		
		new_tokens= []
		mylist1= data[label][category]
		mylist= ["file", "folder", "directory", "contents", "name"]
		mylist.extend(mylist1)
    
		
		for item in sentence_tokens:
			flag=0
			for i in mylist:
				dist= jf.jaro_distance(unicode((item),encoding="utf-8"), unicode(str(i),encoding="utf-8"))
				if dist>=0.8:
					flag=1
			if(flag==0):
				new_tokens.append(item)

		#print("printing new_tokens")
		#print new_tokens

		str1 = ' '.join(new_tokens)

		#print str1



		st = StanfordPOSTagger(config['tagger']['model'],path_to_jar=config['tagger']['path'])
		stanford_tag = st.tag(str1.split())

		#print("Tags")
		#print stanford_tag

    	
    
    
		#print "finding the arguments of the command "
		find_arg(category,stanford_tag)
		if f==1:
			construct_command(category)
		else:
			exit()


	
call_reia()
#create_synonym()
