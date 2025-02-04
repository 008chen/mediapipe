#include "mediapipe/framework/api2/port.h"

#include "mediapipe/framework/port/gtest.h"

namespace mediapipe {
namespace api2 {
namespace {

TEST(PortTest, IntInput) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(port.type_id(), typeid(int).hash_code());
}

TEST(PortTest, OptionalInput) {
  static constexpr auto port = Input<float>::Optional("BAR");
  EXPECT_TRUE(port.IsOptional());
}

TEST(PortTest, Tag) {
  static constexpr auto port = Input<int>("FOO");
  EXPECT_EQ(std::string(port.Tag()), "FOO");
}

struct DeletedCopyType {
  DeletedCopyType(const DeletedCopyType&) = delete;
  DeletedCopyType& operator=(const DeletedCopyType&) = delete;
};

TEST(PortTest, DeletedCopyConstructorInput) {
  static constexpr Input<DeletedCopyType> kInputPort{"INPUT"};
  EXPECT_EQ(std::string(kInputPort.Tag()), "INPUT");

  static constexpr Output<DeletedCopyType> kOutputPort{"OUTPUT"};
  EXPECT_EQ(std::string(kOutputPort.Tag()), "OUTPUT");

  static constexpr SideInput<DeletedCopyType> kSideInputPort{"SIDE_INPUT"};
  EXPECT_EQ(std::string(kSideInputPort.Tag()), "SIDE_INPUT");

  static constexpr SideOutput<DeletedCopyType> kSideOutputPort{"SIDE_OUTPUT"};
  EXPECT_EQ(std::string(kSideOutputPort.Tag()), "SIDE_OUTPUT");
}

}  // namespace
}  // namespace api2
}  // namespace mediapipe
