from scipy.sparse.linalg import svds
from scipy.sparse import diags, csr_matrix
import matplotlib.pyplot as plt
import CosineCluster as CC
from wordcloud import WordCloud
import numpy as np
import matplotlib.cm as cm
import time



def show_heat_map(bins=100):
    (matrix, subreddit_map, reverse_map) = CC.matrix_repre("../data/subreddits")
    matrix = matrix[np.ravel(matrix.sum(axis=1) > 0), :]
    matrix = csr_matrix(matrix, dtype=float)
    matrix = CC.normalize_rows(matrix)
    (u, s, v) = svds(matrix, k=2)
    #pts1 = pts[1:1000, :]
    #plt.scatter(pts1[:, 0], pts1[:, 1])
    #plt.show()
    #pts = u.dot(diags(s, 0)).dot(v)

    heatmap, xedg, yedg = np.histogram2d(pts[:, 0], pts[:, 1], bins=bins)

    heatmap = heatmap[np.ravel(heatmap.sum(axis=1) > 0), :]
    print xedg
    print yedg
    extent = [xedg[0], xedg[-1], yedg[0], yedg[-1]]
    print extent
    plt.clf()
    plt.imshow(heatmap, extent=extent, cmap = cm.Greys_r)
    plt.show()


def rescale_bags(bags, k):
    new_bag = {}
    for label in bags:
        new_bag[label] = {}
        for el in bags[label]:
            new_bag[label][el] = int(k * bags[label][el])
    return new_bag
    
if __name__ == "__main__":
    (matrix, subreddit_map, reverse_map) = CC.matrix_repre("../data/subreddits")
    im = plt.imread("../data/foot.gif")
    im = im[:, :, 0] > 100
    
    matrix = matrix[np.ravel(matrix.sum(axis=1) > 0), :]
    matrix = CC.normalize_rows(matrix)
    labels = CC.cluster_users_matrix(matrix, k=100)
    #bands = [6] * 32
    
    #(labels, centers) = CC.kmeans_LSH(100, matrix,  bands, max_iters=300)
    bags = CC.cluster_centers(matrix, reverse_map, labels)
    key_order = sorted(bags, key=lambda bag_key: sum(labels == bag_key))
    
    re_bags = rescale_bags(bags, 10000)
    for bag_key in key_order:
        print CC.average_sim(matrix[labels == bag_key,:])
        strs = map(lambda x: str(x[0]) + ": %.3f" % x[1], CC.k_representatives(bags[bag_key], 10))
        c_size = sum(labels == bag_key)

        print str(c_size) + " " + " ".join(strs)

        wordcloud = WordCloud(font_path="/Library/Fonts/Verdana.ttf", background_color="white", mask=im)

        wordcloud.fit_words(CC.k_representatives(re_bags[bag_key], 1000))
        wordcloud.to_file("../data/images/image_%d_%d.png"% (c_size, bag_key))

