using OpenCvSharp;
using SAM2_ONNX;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using MemoLibV2.WPF_CustomControl;
using MemoLibV2.WPF_CustomControl.MemoToolEdit;

namespace SAM2_ONNX_UI
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : System.Windows.Window
    {
        SAM2 SamModel;
        public MainWindow()
        {
            
            InitializeComponent();
        }

    }
}
