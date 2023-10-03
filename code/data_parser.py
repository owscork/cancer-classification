import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
"""
File to parse data and scatter plot
"""


def parser(filename):
    

    

    os.chdir(os.getcwd() + "/" + filename)
    #############
    ##
    # p finds prevalence of top two features in each patient
    p = pd.DataFrame(columns = ['hsa-mir-21 reads', 'hsa-mir-30a reads'])
    # q finds the sum of miRNA reads for each each feature in each patient and sorts it high-low
    ##
    #############
    q = pd.DataFrame(columns = ['miRNA_ID', 'reads_per_million_miRNA_mapped'])
    count = 0
    bad_files = ['MANIFEST.txt', 'annotations.txt', '.DS_Store']
    for file in os.listdir():
        if file in bad_files or file.endswith('.png'):
            continue
        else:
            #print(file)
            old_dir = os.getcwd()
            os.chdir(os.getcwd() + "/" + file)
            for file in os.listdir():
                if file in bad_files:
                    continue
                #print(file)
                count += 1
                
                mRNA_prof = pd.read_csv(file, sep="\t")
                p.loc[count] = [mRNA_prof.iloc[285]["reads_per_million_miRNA_mapped"], mRNA_prof.iloc[352]["reads_per_million_miRNA_mapped"]]
                #print(mRNA_prof.shape[0])
                if count != 1:
                    q["miRNA_ID"] = mRNA_prof["miRNA_ID"]
                    q["reads_per_million_miRNA_mapped"] = (mRNA_prof["reads_per_million_miRNA_mapped"] + q["reads_per_million_miRNA_mapped"])
                else:
                    q["miRNA_ID"] = mRNA_prof["miRNA_ID"]
                    q["reads_per_million_miRNA_mapped"] = (mRNA_prof["reads_per_million_miRNA_mapped"])
            os.chdir(old_dir)
    print("------------------------")
    print(mRNA_prof)
    print(q)
    q = q[q["reads_per_million_miRNA_mapped"] > 0]
    q["reads_per_million_miRNA_mapped"] = q["reads_per_million_miRNA_mapped"] / q["reads_per_million_miRNA_mapped"].sum()
    print(q)
    sorted_mRNA = q.sort_values(by=["reads_per_million_miRNA_mapped"], ascending=False)
    print(sorted_mRNA)
    ''''
    plt.scatter(p['hsa-mir-21 reads'], p['hsa-mir-30a reads'])
    plt.xlabel("hsa-mir-21 reads per million")
    plt.ylabel("hsa-mir-30a reads per million")
    plt.axis("square")
    plt.show()
    plt.savefig("Breast Invasive Carcinoma Most Prevalent miRNA")
    '''
    q = q.sort_values(by=["reads_per_million_miRNA_mapped"], ascending=False)
    labels1 = list(q.iloc[0:5]["miRNA_ID"])
    for i in range(len(q["reads_per_million_miRNA_mapped"]) - 5):
        labels1.append('')
    print(len(q["reads_per_million_miRNA_mapped"]))
    def my_autopct(pct):
        return ('%.2f' % pct) if pct > 5 else ''
    plt.pie(q["reads_per_million_miRNA_mapped"],labels=labels1,autopct=my_autopct)
    plt.title('Avg MakeUp')
    print(q.head())
    print(q.iloc[0:5]["reads_per_million_miRNA_mapped"])
    plt.legend(labels1[0:5], loc="lower right") 
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("/Users/owencorkery/Documents/Spring 2022/CSC 371/project3-owen-and-raylon/figs/miRNA features in " + filename)
    plt.clf()
    #plt.show()
    
    #print(count)


##########################################
#replace with folder name of cancer type
# 
os.chdir("/Users/owencorkery/Documents/Spring 2022/CSC 371/project3-owen-and-raylon/data/")
print(os.getcwd())
print(os.listdir())
for file in os.listdir()[1:7]:
    print(file)
    parser(file)
    os.chdir("/Users/owencorkery/Documents/Spring 2022/CSC 371/project3-owen-and-raylon/data/")    
