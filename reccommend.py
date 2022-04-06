import pickle
from collections import Counter
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim.models import TfidfModel
from gensim.corpora.dictionary import Dictionary
from sklearn.neighbors import NearestNeighbors

# Hyperparams. Algorithm behavior will change wildly based on these numbers. num_topics is by far the most important
# The higher num_topics is, the less generic recommendations will be
num_topics = 50  # No idea what this should be really
tfidf_type = None  # Recommended settings: [None, 'nfc']. Having this on weights popular VNs lower. LDA shouldn't need it though.
min_votes_per_user = 5
min_votes_per_vn = 5
good_vote_threshold = 5
lda_iterations = 2000
lda_random_state = 3


def reformat_document(term_and_vote_lst):
    doc = []
    for item in term_and_vote_lst:
        vn_id, vote = item
        if vote >= good_vote_threshold:
            doc += [str(vn_id)] * vote
    return doc


def load_vn_id2title(fname='vn_titles'):
    d = {}
    with open(fname) as f:
        line = f.readline()
        while line:
            vn_id, lang, *rest = line.split()
            vn_id = int(vn_id[1:])
            if lang == 'ja':
                d[vn_id] = ' '.join(rest[:-1]).replace(' \\N', '')
            else:
                if vn_id not in d.keys():
                    d[vn_id] = ' '.join(rest[:-1]).replace(' \\N', '')
            line = f.readline()
    return d


def load_user_id2user_name(fname='users'):
    d = {}
    with open(fname) as f:
        line = f.readline()
        while line:
            user_id, *rest, user_name, _ = line.split()
            user_id = int(user_id[1:])
            d[user_id] = user_name
            line = f.readline()
    return d


def print_topics():
    for topicnum in range(num_topics):
        print('Showing topic {}'.format(topicnum))
        lst = lda.show_topic(topicnum, topn=10)
        print([(gensim_id2title[int(x[0])], x[1]) for x in lst])
        print()


def save_docs(fname):
    def filter_keys(d):
        bad_keys = []
        start_num = len(d.keys())
        for k in d.keys():
            if len(d[k]) < min_votes_per_user:
                bad_keys.append(k)
        for k in bad_keys:
            del d[k]
        print('Removed {} of {} users'.format(len(bad_keys), start_num))
        return d

    print('Reading VNDB db')
    d = {}
    with open(fname) as f:
        line = f.readline()
        while line:
            vn_id, user_id, vote, date = line.split()
            vn_id = int(vn_id)
            user_id = int(user_id)
            vote = int(vote) // 10
            try:
                d[user_id].append((vn_id, vote))
            except KeyError:
                d[user_id] = [(vn_id, vote)]
            line = f.readline()

    d = filter_keys(d)

    with open('user_id2doc.pkl', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)


def make_lda_model(fname='user_id2doc.pkl'):
    print('Making LDA model')
    with open(fname, 'rb') as handle:
        user_id2doc = pickle.load(handle)

    corpus = [reformat_document(item[1]) for item in user_id2doc.items()]

    gensim_id2title = Dictionary(corpus)
    gensim_id2title.filter_extremes(no_below=min_votes_per_vn, no_above=1)
    processed_corpus = [gensim_id2title.doc2bow(text) for text in corpus]

    if tfidf_type:
        print('Training LDA with TFIDF')
        transform = TfidfModel(processed_corpus, dictionary=gensim_id2title, smartirs=tfidf_type)
        transform.save('tfidf')
        processed_corpus = transform[processed_corpus]

    lda = LdaModel(processed_corpus, num_topics=num_topics, iterations=lda_iterations, random_state=lda_random_state)

    lda.save('lda_model')
    gensim_id2title.save('gensim_dict')


def recommend_to_users(user_ids, num_predictions=30):
    output = {'recs': []}
    for user_id in user_ids:
        print('Recommending VNs to user u{}, {}'.format(user_id, user_id2user_name[user_id]))
        output['title'] = 'Recommending VNs to user u{}, {}'.format(user_id, user_id2user_name[user_id])
        print()
        read_vn_ids = [x[0] for x in user_id2doc[user_id]]

        user_document = reformat_document(user_id2doc[user_id])
        unseen_doc = gensim_id2title.doc2bow(user_document)

        if tfidf_type:
            print('Doing inference with TFIDF')
            transform = TfidfModel.load('tfidf')
            unseen_doc = transform[unseen_doc]

        topics_and_probs = lda.get_document_topics(unseen_doc, minimum_probability=-1)
        assert len(topics_and_probs) == num_topics
        sum_of_topic_probs = sum([x[1] for x in topics_and_probs])
        vn_counter = Counter()
        for _ in range(10000):
            drawn_topic = np.random.choice(np.arange(len(topics_and_probs)),
                                           p=[x[1] / sum_of_topic_probs for x in topics_and_probs])
            drawn_topic_words_and_probs = lda.get_topic_terms(drawn_topic, topn=1000)
            words = [x[0] for x in drawn_topic_words_and_probs]
            probs = [x[1] for x in drawn_topic_words_and_probs]
            probs = probs / sum(probs)
            drawn_gensim_id = np.random.choice(words, p=probs)
            vn_counter[gensim_id2title[int(drawn_gensim_id)]] += 1
        vns_and_counts = list(vn_counter.items())
        vns_and_counts.sort(key=lambda x: x[1])
        vns_and_counts.reverse()
        unread_vns = [vn_and_count for vn_and_count in vns_and_counts if int(vn_and_count[0]) not in read_vn_ids]
        for vn_id, count in unread_vns[:num_predictions]:
            print('    {}'.format((vn_id2title[int(vn_id)], count / 10000)))
            output['recs'].append( (vn_id2title[int(vn_id)], count / 10000) )
        print()
        return output


def nneighbors(predict_vector_lst, matrix, num_predictions):
    neigh = NearestNeighbors(n_neighbors=num_predictions + 1)
    neigh.fit(matrix)
    predictions = []

    for predict_vector in predict_vector_lst:
        closest_vectors = neigh.kneighbors(predict_vector)[1]
        closest_vectors = [int(x) for x in np.squeeze(closest_vectors)]
        predictions.append(closest_vectors)
    return predictions


def similar_vns(predict_vn_id_lst, num_predictions=10):
    def vn_id2term_topic_vec(vn_id):
        term_topics_vec = np.zeros((1, num_topics))
        try:
            term_topics = lda.get_term_topics(gensim_id2title.token2id[str(vn_id)], minimum_probability=-1)
            prob_sum = sum([x[1] for x in term_topics])
        except KeyError:  # All votes are hidden or something
            # print('Skipping {}'.format(vn_id2title[vn_id]))
            return term_topics_vec
        for i, prob in term_topics:
            term_topics_vec[0, i] = prob / prob_sum
        return term_topics_vec

    term_topic_matrix = np.zeros((len(vn_id2title.keys()), num_topics))
    for idx, vn_id in enumerate(vn_id2title.keys()):
        term_topic_matrix[idx] = vn_id2term_topic_vec(vn_id)

    closest_vns_lst = nneighbors([vn_id2term_topic_vec(predict_vn_id) for predict_vn_id in predict_vn_id_lst],
                                 term_topic_matrix, num_predictions)
    output = {'recs': []}
    for (predict_vn_id, closest_vns) in zip(predict_vn_id_lst, closest_vns_lst):
        print('Predicting VNs similar to v{}, {}\n'.format(predict_vn_id, vn_id2title[predict_vn_id]))
        output['title'] = 'Predicting VNs similar to v{}, {}\n'.format(predict_vn_id, vn_id2title[predict_vn_id])
        for idx in closest_vns[1:]:
            print('    v{}, {}'.format(list(vn_id2title.keys())[idx],
                                       vn_id2title[list(vn_id2title.keys())[idx]]))

            output['recs'].append( (list(vn_id2title.keys())[idx], vn_id2title[list(vn_id2title.keys())[idx]]) )
        print()
    return output

def similar_users(predict_user_ids, num_predictions=10):
    user_topic_matrix = np.zeros((len(user_id2doc.keys()), num_topics))

    for idx, user_id in enumerate(user_id2doc.keys()):
        user_document = reformat_document(user_id2doc[user_id])
        unseen_doc = gensim_id2title.doc2bow(user_document)

        if tfidf_type:
            print('Doing inference with TFIDF')
            transform = TfidfModel.load('tfidf')
            unseen_doc = transform[unseen_doc]

        user_topic_vec = np.zeros((1, num_topics))
        topics_and_probs = lda.get_document_topics(unseen_doc, minimum_probability=-1)
        sum_of_topic_probs = sum([x[1] for x in topics_and_probs])
        for i, prob in topics_and_probs:
            user_topic_vec[0, i] = prob / sum_of_topic_probs
        user_topic_matrix[idx] = user_topic_vec

    predict_user_idx_lst = [list(user_id2doc.keys()).index(user_id) for user_id in predict_user_ids]
    predict_vector_lst = [np.expand_dims(user_topic_matrix[idx], 0) for idx in predict_user_idx_lst]
    closest_users_idx_lst = nneighbors(predict_vector_lst, user_topic_matrix, num_predictions)

    output = {'recs': []}
    for (predict_user_id, closest_users) in zip(predict_user_ids, closest_users_idx_lst):
        print('Predicting users similar to u{}, {}'.format(predict_user_id, user_id2user_name[predict_user_id]))
        output['title'] = 'Predicting users similar to u{}, {}'.format(predict_user_id, user_id2user_name[predict_user_id])
        for idx in closest_users[1:]:
            print('    u{}, {}'.format(list(user_id2doc.keys())[idx],
                                       user_id2user_name[list(user_id2doc.keys())[idx]]))
            output['recs'].append( (list(user_id2doc.keys())[idx], user_id2user_name[list(user_id2doc.keys())[idx]]) )
        print()
    return output


def initialize(make_lda = False):
    print('WELCOME TO THE GAMBS RECOMMENDATION ENGINE')

    global vn_id2title
    global user_id2user_name
    vn_id2title = load_vn_id2title(fname='vn_titles')
    user_id2user_name = load_user_id2user_name(fname='users')

    # The next two lines only need to be run once
    # Re-run them if you ever change the hyperparameters
    save_docs('vndb-votes-2022-04-06')
    if make_lda:
        make_lda_model()

    global user_id2doc
    global gensim_id2title
    global lda
    with open('user_id2doc.pkl', 'rb') as handle:
        user_id2doc = pickle.load(handle)
    gensim_id2title = Dictionary.load('gensim_dict')
    lda = LdaModel.load('lda_model')


if __name__ == '__main__':
    initialize()

    # users = [196748]

    # recommend_to_users(users)
    # similar_users(users)

    recommend_to_users([195051])
    similar_vns([16150])
    similar_users([195051])
    # print_topics()
