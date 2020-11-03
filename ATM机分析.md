# ATM机分析
## 1、构建出操作的对象
### card类
首先构建一个基础卡类
```c++
class base_card {
public:
	base_card(int n, int m);//构造函数的两个形参
	int get_account_number(void);
	int get_password(void);
	void fu_passsword(int t);

private:
	int account_number;//账号
	int password;//密码
};

```
由于我们所用到的是银行卡，因此我们通过继承来创建银行卡类
```c++
class bank_card :public base_card {
public:
	bank_card(int n, int m, int y);//构造函数，加了一个余额参数
	int get_balances(void);
	void fu_balances(int t);
private:
	int balances;
};
```
此银行卡类公有继承基础卡类，并在基础卡类上扩展了余额
### ATM机操作类
```c++
class ATM {
public:
	void get_in_1(int accont_number, int password);//登录函数
	int F = 1;
	int T = 1;
	int t_1 = 1;
	int r_1 = 1;
private:
	int account_number_b;//从card中获取的账号
	int password_b;//从card中获取的密码
};

```
## 定义操作函数
检验账号及密码函数
```c++
ATM::get_in_1(int accont_number,int password) {       //登录验证函数
	int i = 0;
	int num = 0;
	int num_1 = 0;
	int r = 1,t = 1;
	while (t) {
		for (i = 0; i < 2; i++) {
			//获得原本账号
			account_number_b = obi[i].get_account_number();
			if (account_number_b == accont_number) {
				t = 0;
				t_1 = 0;
				break;
			}
			num++;
		}
		if (num == 2) {
			t = 0;
			r = 0;
		}
	}
	while (r) {
		//获得输入密码和原本密码
		password_b = obi[i].get_password();
		if (password_b == password) {
			r = 0;
			F = 0;
			card = i;
			r_1 = 0;
		}
		else {
			r = 0;
		}
	}
```

## 对于可视化
首先，我们通过添加新的窗口类来添加窗口，至于窗口上所要添加的要素我们要用到一下函数
### 可编辑文本
```c++
void Cchange_password::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Text(pDX, IDC_EDIT1, password_yuan);//定义了一个可编辑文本
}
```
### 按钮
```c++
BEGIN_MESSAGE_MAP(Cselect, CDialogEx)//定义按钮变量
	ON_BN_CLICKED(IDC_BUTTON1, &Cselect::OnBnClickedButton1)
END_MESSAGE_MAP()
```

### 可调整静态文本
```c++
void Cchange_password::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
	DDX_Control(pDX, IDC_TEXT, sss);//定义了可调整静态文本
}

```
### 关闭窗口函数
```c++
CDialogEx::OnOK();
```