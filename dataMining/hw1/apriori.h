#include "association.h"

class Trie
{
public:
    class TrieNode
    {
    public:
        TrieNode(int count) : childs(map<string, TrieNode *>()),
                              count(count){};
        map<string, TrieNode *> childs;
        int count;
    };

    TrieNode *root;

    Trie()
    {
        root = new TrieNode(-1);
    }

    void insert(set<string> transation)
    {
        TrieNode *p = root;
        for (const auto &item : transation)
        {
            if (p->childs[item] == NULL)
            {
                p->childs[item] = new TrieNode(1);
                p = p->childs[item];
            }
            else
            {
                p = p->childs[item];
                p->count++;
            }
        }
    }

    int search(set<string> transation)
    {

        TrieNode *p = root;

        for (const auto &item : transation)
        {
            if (p->childs[item] != NULL)
                p = p->childs[item];
            else
                break;
        }
        return p->count;
    }
};

void outItems(const assoInfo &items)
{

    cout << "=====\n";
    for (const auto &item : items.itemSet)
    {
        cout << item << " ";
    }
    cout << endl;
    for (const auto &p : items.possibleTrans)
    {
        cout << p << " ";
    }
    cout << endl;
    cout << "=====\n";
}

// apriori
vector<assoInfo> apriori(const vector<vector<string>> &data, float minSup)
{
    // return result
    vector<assoInfo> result;

    // dictionary format transations, for example:
    // [{"apple":1, "banana":1, "orange":1}, {"apple":1, "banana":1, "orange":1}]
    vector<unordered_map<string, int>> transations;
    // count base item
    map<string, int> itemCounts;

    // count base item &  save transations for lookup
    for (const vector<string> &t : data)
    {
        unordered_map<string, int> tmap;
        for (const auto &item : t)
        {
            itemCounts[item]++;
            tmap[item]++;
        }
        transations.push_back(tmap);
    }
    vector<string> oneItems;
    int lastidx = 0;

    // the index of item in oneItems, for example:
    // oneItems = ["apple", "banana", "orange"]
    // itemIndex = {"apple":0, "banana":1, "orange":2}
    map<string, int> itemIndex;

    // add freq one item to oneItems
    for (const auto &item : itemCounts)
    {
        if (item.second >= minSup)
        {
            oneItems.push_back(item.first);
            itemIndex[item.first] = lastidx++;
        }
    }

    // generate rule
    queue<assoInfo> itemSets;

    for (const auto &item : oneItems)
    {
        assoInfo asso(set<string>({item}), itemCounts[item]);

        result.push_back(asso); // add one frequent itemset to result

        for (int i = 0; i < transations.size(); i++)
        {
            if (transations[i].count(item))
                asso.possibleTrans.insert(i);
        }
        itemSets.push(asso);
    }

    while (itemSets.size())
    {
        int n = itemSets.size();
        while (n--)
        {
            // get last item in itemset
            auto it = itemSets.front().itemSet.end();

            for (int i = itemIndex[*(--it)] + 1; i < oneItems.size(); i++)
            {
                assoInfo items = itemSets.front();

                items.appearCount = 0;
                items.itemSet.insert(oneItems[i]);

                // scan transations
                auto transIt = items.possibleTrans.begin();
                while (transIt != items.possibleTrans.end())
                {
                    bool notInTrans = false;
                    // loop through all the item in itemSet
                    for (const auto &s : items.itemSet)
                    {
                        if (!transations[*transIt].count(s)) // tras not contain item
                        {
                            notInTrans = true;
                            break;
                        }
                    }
                    if (notInTrans)
                    {
                        transIt = items.possibleTrans.erase(transIt)++;
                    }
                    else
                    {
                        items.appearCount++;
                        ++transIt;
                    }
                }

                if (items.appearCount >= minSup)
                {
                    itemSets.push(items);
                    items.possibleTrans.clear();
                    result.push_back(items);
                }
            }
            itemSets.pop();
        }
    }
    return result;
}
