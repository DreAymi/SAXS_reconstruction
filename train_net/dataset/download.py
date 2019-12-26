import os
str1="http://www.ebi.ac.uk/pdbe/pisa/cgi-bin/multimer.pdb?"
str2=":1,1"
fr=open("interfaces.txt")
arrayOfLine=fr.readlines()
for item in arrayOfLine:
    vec=item.split('?')
    pdbcodeArray=vec[1].split('\n')[0].split(',')
    for pdbcode in pdbcodeArray:
	os.system('wget -O %s.pdb %s%s%s'%(pdbcode,str1,pdbcode,str2))
	fs=open('%s.pdb'%pdbcode)
	word=fs.readlines()
   	print word[0].split(' ')[1]
	if(word[0].split(' ')[1]=='***'):
	    os.system('rm %s.pdb'%pdbcode)
