#include "fpGrowth.h"

// apriori
vector<assoInfo> apriori(const vector<vector<string>> &data, int minSup)
{
    // return result
    vector<assoInfo> result;

    // lookup table of transations
    vector<set<string>> transations;

    // count base item
    map<string, int> oneItemDict;

    // save seen freq itemset
    map<string, int> itemSetHistory;

    // count base item &  save transations
    for (vector<string> const &t : data)
    {
        set<string> temp;
        for (auto item : t)
        {
            oneItemDict[item]++;
            temp.insert(item);
        }
        transations.push_back(temp);
    }

    vector<string> baseItems;
    vector<assoInfo> freqItemSet;

    // add freq one item to baseItems
    for (auto const &item : oneItemDict)
    {
        if (item.second >= minSup)
        {
            baseItems.push_back(item.first);
            assoInfo temp(set<string>({item.first}), item.second);
            freqItemSet.push_back(temp);
            result.push_back(temp);
        }
    }
    int now = 0;
    while (freqItemSet.size())
    {

        vector<assoInfo> currItemSet;
        // previous freq item set

        int prev = 1;
        for (assoInfo const &f : freqItemSet)
        {
            // base item to add

            for (int i = prev++; i < baseItems.size(); i++)
            {
                string oItem = baseItems[i];
                // if already has base item
                if (f.itemSet.count(oItem))
                    continue;

                // add base item
                assoInfo tempAsso = f;
                tempAsso.support = 0;
                tempAsso.itemSet.insert(oItem);

                // check if this pattern exist
                string pattern = "";
                for (auto i : tempAsso.itemSet)
                    pattern += i + ", ";

                if (itemSetHistory.count(pattern))
                    continue;
                else
                    itemSetHistory[pattern] = 1;

                // lookup support from transations
                for (set<string> const &trans : transations)
                {
                    // get union of this freq itemset with transation
                    set<string> ts;
                    // if intersection count == this pattern means it show in transation
                    set_intersection(tempAsso.itemSet.begin(), tempAsso.itemSet.end(), trans.begin(), trans.end(), inserter(ts, ts.begin()));

                    if (ts.size() == tempAsso.itemSet.size())
                    {
                        tempAsso.support++;
                    }
                }
                // delete if support < minup
                if (tempAsso.support < minSup)
                    continue;
                else
                {
                    currItemSet.push_back(tempAsso);
                    result.push_back(tempAsso);
                }
            }
        }
        freqItemSet = currItemSet;
    }

    return result;
}

// get all transations
vector<vector<string>> readData(string filePath)
{
    int id, tranID;
    string item;

    ifstream infile(filePath);

    map<int, vector<string>> datas;

    while (infile >> id >> tranID >> item)
    {
        datas[tranID].push_back(item);
    }

    vector<vector<string>> result;

    for (auto &d : datas)
    {
        result.push_back(d.second);
    }
    return result;
}

// given itemset { 1, 2, 3}
// return all combination  of {} => {} ex. like {1}=>{2,3}
vector<pair<string, string>> allCombination(const set<string> &freqItemSet)
{
    vector<pair<string, string>> result;

    queue<pair<string, string>> que;

    vector<string> itemSet(freqItemSet.begin(), freqItemSet.end());

    que.push(make_pair("", ""));

    int index = 0;

    while (que.size() && index < itemSet.size())
    {

        int n = que.size();
        while (n--)
        {
            auto temp1 = que.front();
            auto temp2 = que.front();
            que.pop();

            temp1.first += itemSet[index] + ", ";
            temp2.second += itemSet[index] + ", ";

            que.push(temp1);
            que.push(temp2);
        }
        index++;
    }
    while (que.size())
    {
        if (que.front().first != "" && que.front().second != "")
            result.push_back(que.front());
        que.pop();
    }
    return result;
}

// output result to file
void printResult(const vector<assoInfo> &freqSet, string setPath, string rulePath, float minSupport, float minConfidence, float transCount)
{
    ofstream myfile;
    myfile.open(setPath);

    int totalCounts = 0;

    map<string, float> ruleCount;

    ofstream rulefile;
    rulefile.open(rulePath);

    // create itemset:freq map
    for (auto const &fSet : freqSet)
    {
        string k = "";
        for (auto const &item : fSet.itemSet)
            k += item + ", ";

        ruleCount[k] = fSet.support;
        rulefile << k << " | " << fSet.support << "\n";
    }
    rulefile.close();

    myfile << " Sup  | Conf  | Rule\n";
    myfile << "====================\n";

    //  confidence filter & print reuslt to file
    for (auto const &itSet : freqSet)
    {
        auto allCombs = allCombination(itSet.itemSet);

        for (auto const &comb : allCombs)
        {

            float sup = float(itSet.support) / transCount;
            float confi = float(itSet.support) / ruleCount[comb.first];

            if (confi < minConfidence)
                continue;

            myfile << setprecision(4) << setw(5) << sup << " | " << setw(5) << confi << " | ";
            myfile << "{ " << comb.first << "} => { " << comb.second << "}\n";

            totalCounts++;
        }
    }
    myfile << "====================\nTotal rules = " << totalCounts << endl;
    myfile.close();
}

int main()
{

    float minSupport = 0.1, confidence = 0.2;

    // read data
    vector<vector<string>> datas = readData("./data.txt");
    /*
    vector<vector<string>> datas = {
        {"milk", "bread", "beer"},
        {"bread", "coffee"},
        {"bread", "egg"},
        {"milk", "bread", "coffee"},
        {"milk", "egg"},
        {"bread", "egg"},
        {"milk", "egg"},
        {"milk", "bread", "egg", "beer"},
        {"milk", "bread", "egg"},
    };*/

    // create tree
    //fpTree tree(int(minSupport * datas.size()));

    //fpTree tree(2);
    // build tree

    int mSup = int(minSupport * datas.size());

    // mSup = 2;

    fpTree tree(mSup);
    tree.buildTree(datas);

    vector<assoInfo> tree_ans = tree.fpMining();
    vector<assoInfo> ap_ans = apriori(datas, mSup);

    printResult(tree_ans, "./fp_result.txt", "fp_rule.txt", minSupport, confidence, datas.size());
    printResult(ap_ans, "./ap_result.txt", "./ap_rule.txt", minSupport, confidence, datas.size());
}