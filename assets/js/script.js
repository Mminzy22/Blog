document.addEventListener('DOMContentLoaded', function () {
    const sidebar = document.querySelector('.sidebar');
    const toggleButton = document.querySelector('.sidebar-toggle');

    // 버튼 클릭 시 사이드바 활성화/비활성화
    toggleButton.addEventListener('click', function () {
        sidebar.classList.toggle('active');
    });

    // 바깥 영역 클릭 시 사이드바 닫기
    document.addEventListener('click', function (event) {
        if (!sidebar.contains(event.target) && !toggleButton.contains(event.target)) {
            sidebar.classList.remove('active');
        }
    });
});
