#include "fpGrowth.hpp"

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

// given itemset { 1, 2, 3}
// return all combination  of {} => {}
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
void printResult(map<string, int> itemFrequency, const vector<assoInfo> &freqSet, string outfile, float minSupport, float minConfidence, float transCount)
{
    ofstream myfile;
    myfile.open(outfile);

    int totalCounts = 0;

    unordered_map<string, float> ruleCount;

    // create itemset:freq map
    for (auto const &fSet : freqSet)
    {
        string k = "";
        for (auto const &item : fSet.itemSet)
            k += item + ", ";

        ruleCount[k] = fSet.support;
    }

    myfile << " Sup  | Conf  | Rule\n";
    myfile << "====================\n";
    // filte with confidence & print reuslt to file
    for (auto const &itSet : freqSet)
    {
        auto allCombs = allCombination(itSet.itemSet);

        for (auto const &comb : allCombs)
        {

            float sup = itSet.support / transCount;
            float confi = itSet.support / ruleCount[comb.first];

            if (confi < minConfidence)
                continue;

            myfile << setprecision(3) << setw(5) << sup << " | " << setw(5) << confi << " | ";
            myfile << "{ " << comb.first << "} => { " << comb.second << "}\n";

            totalCounts++;
        }
    }
    myfile << "====================\nTotal rules = " << totalCounts << endl;
    myfile.close();
}

int main()
{

    float minSupport = 0.4, confidence = 0.2;

    // read data
    vector<vector<string>> datas = readData("./data.txt");

    // create tree
    fpTree tree(int(datas.size() * minSupport));

    // count items
    tree.countItems(datas);

    // create tree
    tree.addNode(datas);

    auto ans = tree.fpMining();

    printResult(tree.frequency, ans, "./fp_result.txt", minSupport, confidence, datas.size());
    int x = 0;
}