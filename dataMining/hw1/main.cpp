#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <fstream>
#include <algorithm>

#include <queue>

using namespace std;

class treeNode
{

public:
    int item;
    treeNode *parent;
    unordered_map<int, treeNode *> childs;
    int count;

    treeNode()
    {
        parent = NULL;
        childs = unordered_map<int, treeNode *>();
        count = -1;
        item = -1;
    }
    treeNode(treeNode *parent, int item)
    {
        parent = parent;
        childs = unordered_map<int, treeNode *>();
        count = 1;
        item = item;
    }
};

class fpTree
{

public:
    treeNode *root;
    unordered_map<int, int> frequency;
    int minSup;
    unordered_map<int, vector<treeNode *>> itemPointers;

    fpTree(int minSup) : minSup(minSup)
    {
        frequency = unordered_map<int, int>({});
        itemPointers = unordered_map<int, vector<treeNode *>>({});
        root = new treeNode();
    }
    // quick sort
    void sortByFreq(vector<int> &transation, int low, int high)
    {
        if (low < high)
        {
            int pi = partition(transation, low, high);
            sortByFreq(transation, low, pi - 1);
            sortByFreq(transation, pi + 1, high);
        }
    }

    int partition(vector<int> &arr, int low, int high)
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
    vector<int> removeUnFreq(const vector<int> &transation)
    {
        vector<int> result;
        for (auto t : transation)
        {
            if (frequency[t] >= minSup)
                result.push_back(t);
        }
        return result;
    }
    void addNode(vector<int> transation)
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
                parent = p;
                treeNode *newItem = new treeNode(parent, item);
                p->childs[item] = newItem;
                itemPointers[item].push_back(newItem);
                p = p->childs[item];
            }
        }
    }

    void countItems(vector<int> transation)
    {
        for (auto &item : transation)
            frequency[item]++;
    }
};

// get all transations
vector<vector<int>> readData(string filePath)
{

    int id, tranID, item;

    ifstream infile(filePath);

    unordered_map<int, vector<int>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<int>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

void showTree(fpTree &tree)
{
    queue<treeNode *> list;

    treeNode *root = tree.root;
    list.push(root);

    while (!list.empty())
    {
        cout << "\n=====================\n";
        int n = list.size();
        while (n--)
        {
            treeNode *curr = list.front();
            list.pop();

            int c = curr->childs.size();
            for (auto item : curr->childs)
            {
                cout << "[" << item.first << "," << item.second->count << "] ";
                list.push(item.second);
            }
        }
    }

    for (auto i : tree.itemPointers)
    {
        cout << i.first << " " << i.second.size() << endl;
    }
}

int main()
{

    treeNode a(NULL, 1);

    // create tree
    fpTree tree(2);

    // read data
    //vector<vector<int>> datas = readData("./data.txt");

    // Milk : 0 Bread:1 Beer : 2 Coffee: 3 Egg : 4
    vector<vector<int>> datas = {
        {0, 1, 2},
        {1, 3},
        {1, 4},
        {0, 1, 3},
        {0, 4},
        {1, 4},
        {0, 4},
        {0, 1, 4, 2},
        {0, 1, 4},
    };

    // count items
    for (auto &d : datas)
        tree.countItems(d);

    // create tree
    for (auto &d : datas)
        tree.addNode(d);

    float confident = 0.5, support = 1;

    showTree(tree);
}