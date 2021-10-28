#include "fpGrowth.h"

// quick sort by freq (high to low)
void fpTree::sortByFreq(vector<string> &transation, int low, int high)
{
    if (low < high)
    {
        int pi = partition(transation, low, high);
        sortByFreq(transation, low, pi - 1);
        sortByFreq(transation, pi + 1, high);
    }
}

int fpTree::partition(vector<string> &arr, int low, int high)
{
    int pivot = frequency[arr[high]];
    int i = low - 1;

    for (int j = low; j < high; j++)
    {
        if (frequency[arr[j]] > pivot)
        {
            i++;
            swap(arr[i], arr[j]);
        }
    }
    swap(arr[i + 1], arr[high]);
    return i + 1;
}

// remove item if it's support < minSupport
vector<string> fpTree::removeUnFreq(const vector<string> &transation)
{
    vector<string> result;
    for (auto t : transation)
    {
        if (frequency.count(t) && frequency[t] >= minSup)
            result.push_back(t);
    }
    return result;
}

// add tree node
void fpTree::addNode(const vector<vector<string>> &transations)
{
    for (auto transation : transations)
    {
        treeNode *p = root;
        treeNode *parent = p;

        transation = removeUnFreq(transation);

        sortByFreq(transation, 0, transation.size() - 1);

        for (auto item : transation)
        {

            if (p->childs.count(item))
            {
                p = p->childs[item];
                p->count++;
            }
            else
            {
                // create new node
                parent = p;
                treeNode *newItem = new treeNode(parent, item);

                // let nextHomonym point to previous item
                if (itemPointers.count(item))
                {
                    newItem->nextHomonym = itemPointers[item];
                }
                itemPointers[item] = newItem;

                // go to child
                p->childs[item] = newItem;
                p = p->childs[item];
            }
        }
    }
}

// sort all item from high freq to low freq & remove unfreq item
void fpTree::createOrder()
{
    vector<string> allItems;

    for (auto const &i : this->frequency)
    {
        string temp = i.first;
        allItems.push_back(temp);
    }

    sortByFreq(allItems, 0, allItems.size() - 1);

    // from low freq to high freq
    reverse(allItems.begin(), allItems.end());
    this->itemOrder = allItems;
}

// build map of < item : frequency> and create order
void fpTree::buildTree(const vector<vector<string>> &datas)
{
    map<string, int> temp;
    for (auto const &transation : datas)
    {
        for (auto const &item : transation)
            temp[item]++;
    }
    for (auto const &i : temp)
    {
        if (i.second >= this->minSup)
        {
            this->frequency[i.first] = i.second;
        }
    }
    createOrder();
    addNode(datas);
}

// find freq itemset
vector<assoInfo> fpTree::fpMining(assoInfo history = assoInfo())
{

    vector<assoInfo> result;

    for (auto item : itemOrder)
    {
        // make a copy of history
        assoInfo newHistory = history;

        // add current item
        newHistory.itemSet.insert(item);
        newHistory.support = this->frequency[item];
        result.push_back(newHistory);

        // create cond tree by item
        treeNode *n = itemPointers[item];
        fpTree cTree = condTree(n);

        // recursive mining
        vector<assoInfo> temp = cTree.fpMining(newHistory);

        // add mingin result to global result
        result.insert(result.end(), temp.begin(), temp.end());
    }
    return result;
}

// create cond tree of item
fpTree fpTree::condTree(treeNode *n)
{

    fpTree tree(minSup);

    vector<vector<string>> newTransations;

    while (n)
    {
        int pCount = n->count;

        vector<string> preNodes;

        treeNode *p = n->parent;

        while (p->parent)
        {
            preNodes.push_back(p->item);
            p = p->parent;
        }
        while (pCount--)
        {
            newTransations.push_back(preNodes);
        }
        n = n->nextHomonym;
    }
    tree.buildTree(newTransations);
    return tree;
}