#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <set>
#include <queue>
#include <iomanip>
#include <unordered_map>

using namespace std;

// record the itemset ans it's support count
class assoInfo
{
public:
    //string item;
    set<string> itemSet;

    int support;

    assoInfo() : itemSet({}), support(0) {}
    assoInfo(const assoInfo &p1)
    {
        //  this->item = p1.item;
        this->itemSet = p1.itemSet;
        this->support = p1.support;
    }
};

class treeNode
{

public:
    string item;
    treeNode *parent;
    map<string, treeNode *> childs;
    int count;
    treeNode *nextHomonym;
    treeNode()
    {
        this->parent = NULL;
        this->count = -1;
        this->item = "root";
        this->nextHomonym = NULL;
    }
    treeNode(treeNode *parent, string item)
    {
        this->parent = parent;
        this->count = 1;
        this->item = item;
        this->nextHomonym = NULL;
    }
};

class fpTree
{
public:
    treeNode *root;
    int minSup;

    map<string, int> frequency;

    // pointer of last seen treeNode of item
    map<string, treeNode *> itemPointers;

    // list all item by freq low to hight
    vector<string> itemOrder;

    // init
    fpTree(int minSup) : minSup(minSup)
    {
        this->frequency = {};
        this->itemPointers = {};
        this->itemOrder = {};
        this->root = new treeNode();
    }

    // quick sort by freq (high to low)
    void sortByFreq(vector<string> &transation, int low, int high);
    int partition(vector<string> &arr, int low, int high);

    // remove item if it's support < minSupport
    vector<string> removeUnFreq(const vector<string> &transation);

    // add tree node
    void addNode(const vector<vector<string>> &transations);

    // sort all item from high freq to low freq & remove unfreq item
    void createOrder();

    // build map of < item : frequency> and create order
    void buildTree(const vector<vector<string>> &datas);

    // find freq itemset
    vector<assoInfo> fpMining(assoInfo history);

    // create cond tree of item
    fpTree condTree(treeNode *n);
};

// show tree structure
void showTree(fpTree &tree)
{
    queue<treeNode *> list;

    treeNode *root = tree.root;
    list.push(root);
    cout << "[root]";
    while (list.size())
    {

        int n = list.size();

        cout << "\n=====================\n";
        while (n--)
        {
            treeNode *curr = list.front();
            list.pop();
            for (auto item : curr->childs)
            {
                cout << "[" << item.first << "," << item.second->count << "] ";
                list.push(item.second);
            }
        }
    }

    for (auto i : tree.itemOrder)
    {
        cout << i << " freq= " << tree.frequency[i] << "\n";
    }
}

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
        if (this->frequency.count(t) && this->frequency[t] >= minSup)
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

        // add mining result to global result
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