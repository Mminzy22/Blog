// 사이드바 관련 DOM 요소 가져오기
const sidebarToggle = document.getElementById('sidebar-toggle'); // 사이드바 열기 버튼
const sidebar = document.getElementById('sidebar'); // 사이드바 요소
const closeSidebar = document.getElementById('close-sidebar'); // 사이드바 닫기 버튼

// 이벤트 핸들러 정의
function openSidebar() {
    sidebar.classList.add('open'); // 사이드바 열기
    document.body.classList.add('sidebar-open'); // 스크롤 방지
    sidebarToggle.classList.add('hidden'); // 열기 버튼 숨기기
    console.log('Sidebar opened'); // 디버깅 로그
}

function closeSidebarHandler() {
    sidebar.classList.remove('open'); // 사이드바 닫기
    document.body.classList.remove('sidebar-open'); // 스크롤 해제
    sidebarToggle.classList.remove('hidden'); // 열기 버튼 다시 표시
    console.log('Sidebar closed'); // 디버깅 로그
}

// 방어 코드: 버튼과 사이드바 요소가 존재할 경우에만 동작
if (sidebarToggle && sidebar && closeSidebar) {
    // 클릭 이벤트 등록
    sidebarToggle.addEventListener('click', openSidebar);
    closeSidebar.addEventListener('click', closeSidebarHandler);

    // 모바일 터치 이벤트 등록 (모바일 환경 대응)
    sidebarToggle.addEventListener('touchstart', openSidebar);
    closeSidebar.addEventListener('touchstart', closeSidebarHandler);

    // 문서 클릭 시 사이드바 외부 클릭 감지
    document.addEventListener('click', (event) => {
        if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
            closeSidebarHandler();
            console.log('Sidebar closed by clicking outside'); // 디버깅 로그
        }
    });

    // 터치 이벤트로 외부 클릭 감지 (모바일 환경 대응)
    document.addEventListener('touchstart', (event) => {
        if (!sidebar.contains(event.target) && !sidebarToggle.contains(event.target)) {
            closeSidebarHandler();
            console.log('Sidebar closed by touching outside'); // 디버깅 로그
        }
    });
} else {
    console.error('Sidebar elements not found'); // 요소가 없을 경우 디버깅 로그
}
