#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/calculators/paddle/paddle_calculator.pb.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"

#include "mediapipe/logxx/testH.h"

namespace mediapipe
{
namespace {

constexpr char kImageTag[] = "IMAGE";
constexpr char kMaxCountTag[] = "MAXCOUNT";
}


  class PaddleCalculator : public CalculatorBase
  {

  public:
    PaddleCalculator(){};

    ~PaddleCalculator(){};

    static ::mediapipe::Status GetContract(CalculatorContract *cc)
    {
      cc->Inputs().Tag(kImageTag).Set<ImageFrame>();
      cc->Outputs().Tag(kImageTag).Set<ImageFrame>();
     
      if (cc->InputSidePackets().HasTag(kMaxCountTag)) {
        cc->InputSidePackets().Tag(kMaxCountTag).Set<int>();
      }
      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Open(CalculatorContext *cc)
    {
      const auto& options = cc->Options<::mediapipe::PaddleCalculatorOptions>();
      auto var = options.option_parameter_1();
      LOG(INFO) << "option_parameter_1:"<<var<<"xx:"<<vvprints();

      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Process(CalculatorContext *cc)
    {
      
      if(!cc->Inputs().Tag(kImageTag).IsEmpty())
      {
        // LOG(INFO) << data_stream;
        // std::unique_ptr<ImageFrame> output_stream = std::make_unique<ImageFrame>(cc->Inputs().Tag(kImageTag).Value());

        const auto& input_img = cc->Inputs().Tag(kImageTag).Get<ImageFrame>();
        cv::Mat input_mat = formats::MatView(&input_img);
        auto output_img = absl::make_unique<ImageFrame>(
        input_img.Format(), input_mat.cols, input_mat.rows);
        cv::Mat output_mat = mediapipe::formats::MatView(output_img.get());

        cv::Point p(200, 200);
        cv::circle(input_mat, p, 100, cv::Scalar(0, 255, 0), -1);
        input_mat.copyTo(output_mat);

        cc->Outputs()
        .Tag(kImageTag)
        .Add(output_img.release(), cc->InputTimestamp());
        // cc->Outputs()
        // .Tag(kImageTag)
        // .AddPacket(cc->Inputs().Tag(kImageTag).Value());
      }
      
      return ::mediapipe::OkStatus();
    }

    ::mediapipe::Status Close(CalculatorContext *cc)
    {

      return ::mediapipe::OkStatus();
    }
  };

  REGISTER_CALCULATOR(PaddleCalculator);

}