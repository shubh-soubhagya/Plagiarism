from datasketch import MinHash


def lsh_similarity(doc1, doc2, num_perm=128):
    doc1_set = set(doc1.split())
    doc2_set = set(doc2.split())

    minhash1 = MinHash(num_perm=num_perm)
    minhash2 = MinHash(num_perm=num_perm)

    for word in doc1_set:
        minhash1.update(word.encode('utf8'))
    for word in doc2_set:
        minhash2.update(word.encode('utf8'))

    return minhash1.jaccard(minhash2)